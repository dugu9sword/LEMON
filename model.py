from buff import *
from typing import NamedTuple
import torch
from buff import focal_loss, group_fields
import math
from frag_enumerator import FragmentEnumerator
from context_enumerator import ContextEnumerator
from seq_encoder import RNNSeqEncoder, FofeSeqEncoder, AverageSeqEncoder
from transformer import TransformerEncoderV2, PositionWiseFeedForward
from functools import lru_cache
from dataset import Sp, match2idx_naive, match2idx_presuff
from token_encoder import BiRNNTokenEncoder, MixEmbedding
from attention import MultiHeadAttention, gen_att_mask
from torch_crf import CRF
from program_args import config


class Luban7(torch.nn.Module):

    def __init__(self,
                 char2idx, bichar2idx, seg2idx, pos2idx, ner2idx,
                 lexicon2idx,
                 label2idx,
                 longest_text_len,
                 ):
        super(Luban7, self).__init__()
        self.char2idx = char2idx
        self.bichar2idx = bichar2idx
        self.seg2idx = seg2idx
        self.label2idx = label2idx
        self.pos2idx = pos2idx

        """ Embedding Layer """
        self.embeds = MixEmbedding(char_vocab_size=len(char2idx),
                                   char_emb_size=config.char_emb_size,
                                   seg_vocab_size=len(seg2idx),
                                   seg_emb_size=config.seg_emb_size,
                                   bichar_vocab_size=len(bichar2idx),
                                   bichar_emb_size=config.bichar_emb_size,
                                   pos_vocab_size=len(pos2idx),
                                   pos_emb_size=config.pos_emb_size,
                                   sparse=config.use_sparse_embed == "on")
        if config.char_emb_size > 0 and config.char_emb_pretrain != 'off':
            load_word2vec(embedding=self.embeds.char_embeds,
                          word2vec_path=config.char_emb_pretrain,
                          norm=True,
                          word_dict=self.char2idx,
                          cached_name="{}.{}.char".format(
                              config.char_emb_pretrain.split('/')[1],
                              config.char_count_gt) if config.load_from_cache == "on" else None
                          )
        if config.bichar_emb_size > 0 and config.bichar_emb_pretrain != 'off':
            load_word2vec(embedding=self.embeds.bichar_embeds,
                          word2vec_path=config.bichar_emb_pretrain,
                          norm=True,
                          word_dict=self.bichar2idx,
                          cached_name="{}.{}.bichar".format(
                              config.bichar_emb_pretrain.split('/')[1],
                              config.bichar_count_gt) if config.load_from_cache == "on" else None
                          )
        self.embeds.show_mean_std()

        embed_dim = self.embeds.embedding_dim

        if config.token_type == "tfer":
            self.token_encoder = TransformerEncoderV2(
                d_model=embed_dim,
                len_max_seq=longest_text_len,
                n_layers=config.tfer_num_layer,
                n_head=config.tfer_num_head,
                d_head=config.tfer_head_dim,
                dropout=config.drop_token_encoder
            )
            token_dim = embed_dim
        elif config.token_type == "rnn":
            self.token_encoder = BiRNNTokenEncoder(
                cell_type='lstm',
                num_layers=config.rnn_num_layer,
                input_size=embed_dim,
                hidden_size=config.rnn_hidden,
                dropout=config.drop_token_encoder
            )
            token_dim = config.rnn_hidden
        elif config.token_type == 'plain':
            token_dim = embed_dim
        else:
            raise Exception

        self.ner_score = torch.nn.Sequential(
            torch.nn.Linear(token_dim, token_dim * 2),
            torch.nn.ReLU(),
            torch.nn.Linear(token_dim * 2, len(ner2idx)),
        )
        self.ner_crf = CRF(len(ner2idx), batch_first=True)

        """ Fragment & Context Layer"""
        frag_dim = 0
        if config.frag_type == "rnn":
            self.fragment_encoder = FragmentEnumerator(
                max_span_len=config.max_span_length,
                encoder_cls=RNNSeqEncoder,
                encoder_args=('lstm', token_dim, token_dim),
                fusion=config.frag_fusion
            )
        elif config.frag_type == "fofe":
            self.fragment_encoder = FragmentEnumerator(
                max_span_len=config.max_span_length,
                encoder_cls=FofeSeqEncoder,
                encoder_args=(config.frag_fofe_alpha,),
                fusion=config.frag_fusion
            )
        elif config.frag_type == "average":
            self.fragment_encoder = FragmentEnumerator(
                max_span_len=config.max_span_length,
                encoder_cls=AverageSeqEncoder,
                encoder_args=(),
                fusion=config.frag_fusion
            )
        elif config.frag_type == "off":
            pass
        else:
            raise Exception

        if config.frag_type != "off":
            if config.frag_fusion == 'cat':
                frag_dim += 2 * token_dim
            elif config.frag_fusion == 'add':
                frag_dim += token_dim
            else:
                raise Exception

            if config.frag_att_type != "off":
                self.multi_att = MultiHeadAttention(
                    d_q=frag_dim, d_k=token_dim, d_v=token_dim, d_out=frag_dim,
                    d_att_k=frag_dim // config.frag_att_head,
                    d_att_v=frag_dim // config.frag_att_head,
                    n_head=config.frag_att_head,
                    dropout=config.drop_default
                )

                self.att_norm = torch.nn.LayerNorm(frag_dim)
                frag_dim += {
                    "cat": frag_dim, "add": 0
                }[config.frag_att_type]

        if config.ctx_type in ['include', 'exclude']:
            self.context_encoder = ContextEnumerator(
                max_span_len=config.max_span_length,
                encoder_cls=RNNSeqEncoder,
                encoder_args=('lstm', token_dim, token_dim),
                out_size=token_dim,
                include=config.ctx_type == 'include'
            )
            frag_dim += token_dim + token_dim

        """ Non Linear Stack """
        self.non_linear_stack = torch.nn.ModuleList([
            NonLinearLayerWithRes(frag_dim, 2 * frag_dim, dropout=config.drop_nonlinear)
            for _ in range(config.num_nonlinear)
        ])

        """ Lexicon Embedding """
        if config.match_mode != "off":
            self.lexicon_embeds = torch.nn.Embedding(len(lexicon2idx),
                                                     config.lexicon_emb_dim,
                                                     sparse=config.use_sparse_embed == "on")
            load_word2vec(
                self.lexicon_embeds,
                lexicon2idx,
                config.lexicon_emb_pretrain,
                norm=True,
                cached_name="{}.lexicon".format(
                    config.lexicon_emb_pretrain.split('/')[1]) if config.load_from_cache == "on" else None
            )
        if config.match_mode == "naive":
            self.match_embeds = torch.nn.Embedding(len(match2idx_naive),
                                                   config.match_emb_size,
                                                   sparse=config.use_sparse_embed == "on")
            self.lexicon_attention = MultiHeadAttention(d_q=frag_dim,
                                                        d_k=50 + config.match_emb_size,
                                                        d_v=50 + config.match_emb_size,
                                                        d_att_k=frag_dim // 2,
                                                        d_att_v=frag_dim // 2,
                                                        n_head=2,
                                                        dropout=config.drop_default,
                                                        d_out=frag_dim)
            frag_dim = frag_dim * 2
        elif config.match_mode == "presuff":
            self.match_embeds = torch.nn.Embedding(len(match2idx_presuff),
                                                   config.match_emb_size,
                                                   sparse=config.use_sparse_embed == "on")
            frag_dim = frag_dim + config.lexicon_emb_dim * 2 + config.match_emb_size
        else:
            raise Exception

        self.label_weight = torch.nn.Parameter(torch.Tensor(frag_dim, len(label2idx)))
        self.label_bias = torch.nn.Parameter(torch.Tensor(len(label2idx)))

        std = 1. / math.sqrt(self.label_weight.size(1))
        self.label_weight.data.uniform_(-std, std)
        self.label_bias.data.uniform_(-std, std)

    @property
    def device(self):
        return next(self.parameters()).device

    def gen_span_ys(self, texts, labels):
        text_lens = batch_lens(texts)
        span_ys = []
        for bid in range(len(texts)):
            for begin_token_idx in range(text_lens[bid]):
                for span_len in range(1, config.max_span_length + 1):
                    end_token_idx = begin_token_idx + span_len - 1
                    if end_token_idx < text_lens[bid]:
                        has_label = False
                        for span in labels[bid]:
                            if span.b == begin_token_idx and span.e == end_token_idx:
                                span_ys.append(span.y)
                                has_label = True
                                break
                        if not has_label:
                            span_ys.append(0)
                        # log(begin_token_idx, ", ", end_token_idx, " of ", text_lens[bid])
        return span_ys

    def compute_token_reprs(self, batch_data):
        chars, bichars, segs, poss = group_fields(
            batch_data,
            keys=["chars", "bichars", "segs", "poss"]
        )
        text_lens = batch_lens(chars)

        pad_chars = batch_pad(chars, self.char2idx[Sp.pad])
        pad_bichars = batch_pad(bichars, self.bichar2idx[Sp.pad])
        pad_segs = batch_pad(segs, self.seg2idx[Sp.pad])
        pad_poss = batch_pad(poss, self.pos2idx[Sp.pad])

        pad_chars_tensor = torch.tensor(pad_chars, device=self.device)
        pad_bichars_tensor = torch.tensor(pad_bichars, device=self.device)
        pad_segs_tensor = torch.tensor(pad_segs, device=self.device)
        pad_poss_tensor = torch.tensor(pad_poss, device=self.device)

        input_embs = self.embeds(pad_chars_tensor,
                                 pad_bichars_tensor,
                                 pad_segs_tensor,
                                 pad_poss_tensor)

        if config.token_type == 'rnn':
            token_reprs = self.token_encoder(input_embs, text_lens)
        elif config.token_type == 'tfer':
            masks = self.token_encoder.gen_masks(pad_chars_tensor)
            token_reprs = self.token_encoder(input_embs, masks, text_lens)
        else:
            token_reprs = input_embs
        return token_reprs

    def crf_nll(self, batch_data) -> torch.Tensor:
        gold_tags = group_fields(batch_data, keys="ners")
        token_reprs = self.compute_token_reprs(batch_data)
        scores = self.ner_score(token_reprs)
        gold_tags = torch.tensor(batch_pad(gold_tags, 0))
        masks = torch.tensor(batch_mask(gold_tags, mask_zero=True), dtype=torch.uint8, device=self.device)
        crf_loss = self.ner_crf(scores, gold_tags, masks, reduction="mean")
        return - crf_loss

    def crf_decode(self, batch_data) -> List[List[int]]:
        chars = group_fields(batch_data, keys="chars")
        token_reprs = self.compute_token_reprs(batch_data)
        scores = self.ner_score(token_reprs)
        masks = torch.tensor(batch_mask(chars, mask_zero=True), dtype=torch.uint8, device=self.device)
        results = self.ner_crf.decode(scores, masks)
        return results

    def get_span_score_tags(self, batch_data):
        chars = group_fields(batch_data, keys='chars')
        text_lens = batch_lens(chars)
        labels = group_fields(batch_data, keys='labels')
        token_reprs = self.compute_token_reprs(batch_data)

        if config.frag_type != "off":
            frag_reprs = self.fragment_encoder(token_reprs, text_lens)
            if config.frag_att_type != "off":
                # print(Color.red("ATTENTION!"))
                d_frag = frag_reprs.size(1)
                att_frag_reprs = []
                offset = 0
                for i, text_len in enumerate(text_lens):
                    q = frag_reprs[offset: offset + span_num(text_len)].unsqueeze(0)
                    k = token_reprs[i][:text_len].unsqueeze(0)
                    v = k
                    mask = gen_att_mask(k_lens=(text_len,),
                                        max_k_len=text_len, max_q_len=q.size(1)).to(self.device)
                    att_frag_repr, att_score = self.multi_att(q, k, v, mask)
                    att_frag_reprs.append(att_frag_repr.squeeze(0))
                    offset += text_len
                att_frag_reprs = torch.cat(att_frag_reprs)

                att_frag_reprs = self.att_norm(att_frag_reprs) / math.sqrt(d_frag)
                show_mean_std(frag_reprs)
                show_mean_std(att_frag_reprs)
                if config.frag_att_type == 'cat':
                    frag_reprs = torch.cat([frag_reprs, att_frag_reprs], dim=1)
                elif config.frag_att_type == 'add':
                    frag_reprs = (frag_reprs + att_frag_reprs) / math.sqrt(2)
                else:
                    raise Exception
        else:
            frag_reprs = None

        if config.ctx_type in ['include', 'exclude']:
            left_ctx_reprs, right_ctx_reprs = self.context_encoder(token_reprs, text_lens)
            if frag_reprs is not None:
                frag_reprs = torch.cat([frag_reprs, left_ctx_reprs, right_ctx_reprs], dim=1)
            else:
                frag_reprs = torch.cat([left_ctx_reprs, right_ctx_reprs], dim=1)
        elif config.ctx_type == 'off':
            pass
        else:
            raise Exception

        if config.match_mode == "naive":
            lexmatches = group_fields(batch_data, keys='lexmatches')
            lexmatches = [item[1] for sublist in lexmatches for item in sublist]
            assert len(lexmatches) == frag_reprs.size(0)
            frag_match_lexicons, frag_match_types = [], []
            for it_match in lexmatches:
                if len(it_match) == 0:
                    frag_match_lexicons.append([])
                    frag_match_types.append([])
                else:
                    frag_match_lexicons.append(group_fields(it_match, indices=0))
                    frag_match_types.append(group_fields(it_match, indices=1))

            mask = gen_att_mask(batch_lens(frag_match_lexicons), 8, 1).to(self.device)
            frag_match_lexicons = batch_pad(frag_match_lexicons, pad_len=8)
            frag_match_types = batch_pad(frag_match_types, pad_len=8)
            frag_match_lexicons = torch.tensor(frag_match_lexicons, dtype=torch.long, device=self.device)
            frag_match_types = torch.tensor(frag_match_types, dtype=torch.long, device=self.device)
            mem_lexicon = self.lexicon_embeds(frag_match_lexicons)
            mem_match = self.match_embeds(frag_match_types)
            memory = torch.cat([mem_lexicon, mem_match], dim=2)
            att_word, _ = self.lexicon_attention(frag_reprs.unsqueeze(1), memory, memory, mask)
            frag_reprs = torch.cat([frag_reprs, att_word.squeeze(1)], dim=1)
        elif config.match_mode == "presuff":
            lexmatches = group_fields(batch_data, keys='lexmatches')
            pre_match_idx = [item[1] for sublist in lexmatches for item in sublist]
            suff_match_idx = [item[2] for sublist in lexmatches for item in sublist]
            match_type_idx = [item[3] for sublist in lexmatches for item in sublist]
            assert len(pre_match_idx) == len(suff_match_idx) == len(match_type_idx) == frag_reprs.size(0)
            pre_match_lex = self.lexicon_embeds(torch.tensor(pre_match_idx, dtype=torch.long, device=self.device))
            suff_match_lex = self.lexicon_embeds(torch.tensor(suff_match_idx, dtype=torch.long, device=self.device))
            match_type = self.match_embeds(torch.tensor(match_type_idx, dtype=torch.long, device=self.device))
            frag_reprs = torch.cat([frag_reprs, pre_match_lex, suff_match_lex, match_type], dim=1)
        else:
            raise Exception

        span_ys = self.gen_span_ys(chars, labels)
        score = frag_reprs @ self.label_weight + self.label_bias
        return score, span_ys


@lru_cache(maxsize=None)
def span_num(sentence_length):
    if sentence_length > config.max_span_length:
        return int((sentence_length + sentence_length - config.max_span_length + 1) * config.max_span_length / 2)
    else:
        return int((sentence_length + 1) * sentence_length / 2)
