# -*- coding: UTF-8 -*-

from buff import *
from typing import NamedTuple
from torch.nn.utils import clip_grad_norm_
import torch
from buff import focal_loss
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import os
import math
from collections import defaultdict
from frag_enumerator import FragmentEnumerator
from context_enumerator import ContextEnumerator
from seq_encoder import BaseSeqEncoder, RNNSeqEncoder, FofeSeqEncoder, AverageSeqEncoder
import socket
from transformer import TransformerEncoderV2, PositionWiseFeedForward
from functools import lru_cache
from dataset_v2 import ConllDataSet, SpanLabel, SpanPred, load_vocab, gen_vocab, usable_data_sets
from token_encoder import BiRNNTokenEncoder, MixEmbedding


class ProgramArgs(argparse.Namespace):
    def __init__(self):
        super(ProgramArgs, self).__init__()
        self.max_span_length = 10
        self.max_sentence_length = 120
        self.char_count_gt = 2
        self.bichar_count_gt = 2

        self.token_type = "rnn"

        # embedding settings
        self.char_emb_size = 50
        self.bichar_emb_size = 0
        self.seg_emb_size = 25
        self.pos_emb_size = 25
        self.pos_bmes = 'off'
        # self.char_emb_pretrain = "word2vec/lattice_lstm/gigaword_chn.all.a2b.uni.ite50.vec"
        # self.char_emb_pretrain = "word2vec/sgns/sgns.merge.char"
        # self.char_emb_pretrain = "word2vec/fasttext/wiki.zh.vec"
        self.char_emb_pretrain = "off"

        # self.bichar_emb_pretrain = "word2vec/lattice_lstm/gigaword_chn.all.a2b.bi.ite50.vec"
        self.bichar_emb_pretrain = 'off'

        # transformer config
        self.tfer_num_layer = 2
        self.tfer_num_head = 1
        self.tfer_head_dim = 128

        # rnn config
        self.rnn_num_layer = 2
        self.rnn_hidden = 256

        # fragment encoder
        self.frag_type = "rnn"
        self.frag_fofe_alpha = 0.5

        # context encoder
        # self.ctx = 'off'
        self.ctx = 'off'

        # loss
        self.focal_gamma = 0
        self.focal_reduction = "mean"

        # regularization
        self.drop_embed = 0.1
        self.drop_token = 0.1
        # self.drop_frag = 0.1
        self.drop_nonlinear = 0.1
        self.weight_decay = 1e-5

        # development config
        self.batch_size = 32
        self.load_from_cache = False
        self.train_on = True
        self.use_data_set = "full"
        self.epoch_max = 30
        self.epoch_show_train = 60
        self.model_name = "off"
        self.model_ckpt = -1


parser = argparse.ArgumentParser()
nsp = ProgramArgs()
for key, value in nsp.__dict__.items():
    parser.add_argument('--{}'.format(key),
                        action='store',
                        default=value,
                        type=type(value),
                        dest=str(key))
config = parser.parse_args(namespace=nsp)  # type: ProgramArgs

# if socket.gethostname() == "matrimax":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# elif socket.gethostname() == "localhost.localdomain":
#     os.environ["CUDA_VISIBLE_DEVICES"] = "2"

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")


class NonLinearLayer(torch.nn.Module):
    def __init__(self, d_in, d_hidden, dropout):
        super(NonLinearLayer, self).__init__()
        self.fc1 = torch.nn.Linear(d_in, d_hidden)
        self.fc2 = torch.nn.Linear(d_hidden, d_in)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x)))
        out += x
        out = self.drop(out)
        # out = torch.nn.LayerNorm(out)
        return out


class Luban7(torch.nn.Module):
    def __init__(self,
                 char2idx, bichar2idx, seg2idx, pos2idx,
                 label2idx,
                 longest_text_len,
                 ):
        super(Luban7, self).__init__()
        self.char2idx = char2idx
        self.bichar2idx = bichar2idx
        self.seg2idx = seg2idx
        self.label2idx = label2idx
        self.pos2idx = pos2idx

        self.embeds = MixEmbedding(char_vocab_size=len(char2idx),
                                   char_emb_size=config.char_emb_size,
                                   seg_vocab_size=len(seg2idx),
                                   seg_emb_size=config.seg_emb_size,
                                   bichar_vocab_size=len(bichar2idx),
                                   bichar_emb_size=config.bichar_emb_size,
                                   pos_vocab_size=len(pos2idx),
                                   pos_emb_size=config.pos_emb_size)
        if config.char_emb_size > 0 and config.char_emb_pretrain != 'off':
            load_word2vec(embedding=self.embeds.char_embeds,
                          word2vec_path=config.char_emb_pretrain,
                          norm=True,
                          word_dict=self.char2idx,
                          cached_name="{}.{}.char".format(
                              config.char_emb_pretrain.split('/')[1],
                              config.char_count_gt)
                          )
        if config.bichar_emb_size > 0 and config.bichar_emb_pretrain != 'off':
            load_word2vec(embedding=self.embeds.bichar_embeds,
                          word2vec_path=config.bichar_emb_pretrain,
                          norm=True,
                          word_dict=self.bichar2idx,
                          cached_name="{}.{}.bichar".format(
                              config.bichar_emb_pretrain.split('/')[1],
                              config.bichar_count_gt)
                          )
        torch.nn.init.normal_(self.embeds.seg_embeds.weight, 0,
                              torch.std(self.embeds.char_embeds.weight).item())
        self.embeds.show_mean_std()

        if config.token_type == "tfer":
            self.token_encoder = TransformerEncoderV2(
                d_model=self.embeds.embedding_dim,
                len_max_seq=longest_text_len,
                n_layers=config.tfer_num_layer,
                n_head=config.tfer_num_head,
                d_head=config.tfer_head_dim,
                dropout=config.drop_token
            )
            token_dim = self.embeds.embedding_dim
        elif config.token_type == "rnn":
            self.token_encoder = BiRNNTokenEncoder(
                cell_type='lstm',
                num_layers=config.rnn_num_layer,
                input_size=self.embeds.embedding_dim,
                hidden_size=config.rnn_hidden,
                dropout=config.drop_token
            )
            token_dim = config.rnn_hidden
        elif config.token_type == 'plain':
            token_dim = self.embeds.embedding_dim
        else:
            raise Exception

        if config.frag_type == "rnn":
            self.fragment_encoder = FragmentEnumerator(
                max_span_len=config.max_span_length,
                encoder_cls=RNNSeqEncoder,
                encoder_args=('lstm', token_dim, token_dim),

            )
            if config.ctx in ['include', 'exclude']:
                self.context_encoder = ContextEnumerator(
                    max_span_len=config.max_span_length,
                    encoder_cls=RNNSeqEncoder,
                    encoder_args=('lstm', token_dim, token_dim),
                    out_size=token_dim,
                    include=config.ctx == 'include'
                )
        elif config.frag_type == "fofe":
            self.fragment_encoder = FragmentEnumerator(
                max_span_len=config.max_span_length,
                encoder_cls=FofeSeqEncoder,
                encoder_args=(config.frag_fofe_alpha,)
            )
        elif config.frag_type == "average":
            self.fragment_encoder = FragmentEnumerator(
                max_span_len=config.max_span_length,
                encoder_cls=AverageSeqEncoder,
                encoder_args=(),
            )
        else:
            raise Exception

        frag_dim = 2 * token_dim
        if config.ctx in ['include', 'exclude']:
            frag_dim += token_dim + token_dim

        self.non_linear = NonLinearLayer(frag_dim, 2 * frag_dim, dropout=config.drop_nonlinear)

        self.label_weight = torch.nn.Parameter(torch.Tensor(frag_dim, len(label2idx)))
        self.label_bias = torch.nn.Parameter(torch.Tensor(len(label2idx)))

        std = 1. / math.sqrt(self.label_weight.size(1))
        self.label_weight.data.uniform_(-std, std)
        self.label_bias.data.uniform_(-std, std)

    @staticmethod
    def gen_span_ys(texts, labels):
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

    def forward(self, batch_data):
        chars = list(map(lambda x: x[0], batch_data))
        bichars = list(map(lambda x: x[1], batch_data))
        segs = list(map(lambda x: x[2], batch_data))
        poss = list(map(lambda x: x[3], batch_data))
        labels = list(map(lambda x: x[4], batch_data))
        text_lens = batch_lens(chars)

        pad_chars = batch_pad(chars, self.char2idx['<PAD>'])
        pad_bichars = batch_pad(bichars, self.bichar2idx['<PAD>'])
        pad_segs = batch_pad(segs, self.seg2idx['<PAD>'])
        pad_poss = batch_pad(poss, self.pos2idx['<PAD>'])

        pad_chars_tensor = torch.tensor(pad_chars).to(device)
        pad_bichars_tensor = torch.tensor(pad_bichars).to(device)
        pad_segs_tensor = torch.tensor(pad_segs).to(device)
        pad_poss_tensor = torch.tensor(pad_poss).to(device)

        input_embs = self.embeds(pad_chars_tensor,
                                 pad_bichars_tensor,
                                 pad_segs_tensor,
                                 pad_poss_tensor)

        if config.token_type == 'rnn':
            token_repr = self.token_encoder(input_embs, text_lens)
        elif config.token_type == 'tfer':
            masks = self.token_encoder.gen_masks(pad_chars_tensor)
            token_repr = self.token_encoder(input_embs, masks, text_lens)
        else:
            token_repr = input_embs

        frag_reprs = self.fragment_encoder(token_repr, text_lens)

        if config.ctx in ['include', 'exclude']:
            left_ctx_reprs, right_ctx_reprs = self.context_encoder(token_repr, text_lens)
            frag_reprs = torch.cat([frag_reprs, left_ctx_reprs, right_ctx_reprs], dim=1)
        elif config.ctx == 'off':
            pass
        else:
            raise Exception

        frag_reprs = self.non_linear(frag_reprs)

        span_ys = self.gen_span_ys(chars, labels)
        score = frag_reprs @ self.label_weight + self.label_bias
        return score, span_ys


@lru_cache(maxsize=None)
def span_num(sentence_length):
    if sentence_length > config.max_span_length:
        return (sentence_length + sentence_length - config.max_span_length + 1) * config.max_span_length / 2
    else:
        return (sentence_length + 1) * sentence_length / 2


@lru_cache(maxsize=None)
def enum_span_by_length(text_len):
    span_lst = []
    for begin_token_idx in range(text_len):
        for span_len in range(1, config.max_span_length + 1):
            end_token_idx = begin_token_idx + span_len - 1
            if end_token_idx < text_len:
                span_lst.append((begin_token_idx, end_token_idx))  # tuple
    return span_lst


def main():
    log_config("main.txt", "cf")
    for key, value in nsp.__dict__.items():
        log("\t--{}={}".format(key, value))
    used_data_set = usable_data_sets[config.use_data_set]
    vocab_folder = "dataset/ontonotes4/vocab.{}.{}.{}".format(
        config.char_count_gt, config.bichar_count_gt, config.pos_bmes)
    gen_vocab(data_path=used_data_set[0],
              out_folder=vocab_folder,
              char_count_gt=config.char_count_gt,
              bichar_count_gt=config.bichar_count_gt,
              use_cache=config.load_from_cache,
              ignore_tag_bmes=config.pos_bmes == 'off')
    char2idx, idx2char = load_vocab("{}/char.vocab".format(vocab_folder))
    bichar2idx, idx2bichar = load_vocab("{}/bichar.vocab".format(vocab_folder))
    seg2idx, idx2seg = load_vocab("{}/seg.vocab".format(vocab_folder))
    pos2idx, idx2pos = load_vocab("{}/pos.vocab".format(vocab_folder))
    label2idx, idx2label = load_vocab("{}/label.vocab".format(vocab_folder))

    idx2str = lambda idx_lst: "".join(map(lambda x: idx2char[x], idx_lst))
    train_set = ConllDataSet(data_path=used_data_set[0],
                             char2idx=char2idx, bichar2idx=bichar2idx,
                             seg2idx=seg2idx, label2idx=label2idx, pos2idx=pos2idx,
                             ignore_pos_bmes=config.pos_bmes == 'off',
                             max_text_len=config.max_sentence_length,
                             max_span_len=config.max_span_length,
                             sort_by_length=True)
    dev_set = ConllDataSet(data_path=used_data_set[1],
                           char2idx=char2idx, bichar2idx=bichar2idx,
                           seg2idx=seg2idx, label2idx=label2idx, pos2idx=pos2idx,
                           ignore_pos_bmes=config.pos_bmes == 'off',
                           sort_by_length=False)
    longest_span_len = max(train_set.longest_span_len, dev_set.longest_span_len)
    longest_text_len = max(train_set.longest_text_len, dev_set.longest_text_len)

    luban7 = Luban7(char2idx=char2idx,
                    bichar2idx=bichar2idx,
                    seg2idx=seg2idx,
                    pos2idx=pos2idx,
                    label2idx=label2idx,
                    longest_text_len=longest_text_len).to(device)
    opt = torch.optim.Adam(luban7.parameters(), lr=0.001, weight_decay=config.weight_decay)
    manager = ModelManager(luban7, config.model_name, init_ckpt=config.model_ckpt) \
        if config.model_name != "off" else None
    # opt = torch.optim.SGD(luban7.parameters(), lr=0.1, momentum=0.9)

    epoch_id = -1
    while True:
        epoch_id += 1
        if epoch_id == config.epoch_max:
            break

        """
        Training
        """
        if config.train_on:
            log(">>> epoch {} train".format(epoch_id))
            luban7.train()
            train_set.reset(shuffle=True)
            iter_id = 0
            progress = ProgressManager(total=train_set.size)
            log(train_set.size)
            while not train_set.finished:
                iter_id += 1
                batch_data = train_set.next_batch(config.batch_size)
                batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)

                score, span_ys = luban7(batch_data)

                loss = focal_loss(inputs=score,
                                  targets=torch.tensor(span_ys).to(device),
                                  gamma=config.focal_gamma)
                score_probs = F.softmax(score, dim=1)

                pred = cast_list(torch.argmax(score, 1).cpu().numpy())

                progress.update(len(batch_data))
                if iter_id % 1 == 0:
                    log("".join([
                        "[{}: {}/{}] ".format(epoch_id, progress.complete_num, train_set.size),
                        "b: {:.4f} / c:{:.4f} / r: {:.4f} "
                            .format(progress.batch_time, progress.cost_time, progress.rest_time),
                        "loss: {:.4f} ".format(loss.item()),
                        "accuracy: {:.4f}".format(accuracy(score_probs.detach().cpu().numpy(), span_ys))])
                    )
                opt.zero_grad()
                loss.backward()
                clip_grad_norm_(luban7.parameters(), 5)
                opt.step()
            log("<<< epoch {} train".format(epoch_id))

        if isinstance(manager, ModelManager):
            manager.save()

        """
        Development
        """
        with torch.no_grad():
            luban7.eval()
            sets_for_validation = {"dev_set": dev_set}
            if epoch_id > config.epoch_show_train:
                sets_for_validation["train_set"] = train_set
            for set_name, set_for_validation in sets_for_validation.items():
                log(">>> epoch {} validation on {}".format(epoch_id, set_name))

                set_for_validation.reset(shuffle=False)
                progress = ProgressManager(total=set_for_validation.size)
                while not set_for_validation.finished:
                    batch_data = set_for_validation.next_batch(10, fill_batch=True)
                    batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)  # type: List[Datum]

                    texts = list(map(lambda x: x[0], batch_data))
                    text_lens = batch_lens(texts)

                    score, span_ys = luban7(batch_data)
                    score_probs = F.softmax(score, dim=1)

                    pred = cast_list(torch.argmax(score, 1))

                    offset = 0
                    to_log = []
                    for bid in range(len(text_lens)):
                        to_log.append("[{:>4}] {}".format(
                            progress.complete_num + bid,
                            idx2str(batch_data[bid].chars)))
                        enum_spans = enum_span_by_length(text_lens[bid])
                        # fragment_score = score[offset: offset + len(enum_spans)]
                        # log(fragment_score)
                        for sid, span in enumerate(enum_spans):
                            begin_idx, end_idx = span
                            span_offset = sid + offset
                            if pred[span_offset] != 0 or span_ys[span_offset] != 0:
                                to_log.append("{:>3}~{:<3}\t{:1}\t{:.4f}/{:.4f}\t{:>5}/{:<5}\t{}".format(
                                    begin_idx, end_idx,
                                    pred[span_offset],
                                    score_probs[span_offset][pred[span_offset]],
                                    score_probs[span_offset][span_ys[span_offset]],
                                    idx2label[pred[span_offset]],
                                    idx2label[span_ys[span_offset]],
                                    idx2str(batch_data[bid].chars[begin_idx: end_idx + 1]),
                                ))
                        offset += len(enum_spans)
                    log("\n".join(to_log))
                    progress.update(len(batch_data))

                log("<<< epoch {} validation on {}".format(epoch_id, set_name))


if __name__ == '__main__':
    main()
