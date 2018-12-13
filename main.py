# -*- coding: UTF-8 -*-

from buff import *
from typing import NamedTuple
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import os
import math
from collections import defaultdict
from fragment_encoder import AllFragEncoder, RNNFragEncoder, FofeFragEncoder, AverageFragEncoder
import socket
from transformer import TransformerEncoderV2, PositionWiseFeedForward
from functools import lru_cache
from dataset import ConllDataSet, SpanLabel, SpanPred, load_vocab, gen_vocab, usable_data_sets
from token_encoder import BiRNNTokenEncoder, MixEmbedding


class ProgramArgs(argparse.Namespace):
    def __init__(self):
        super(ProgramArgs, self).__init__()
        self.max_span_length = 10
        self.max_sentence_length = 120
        self.char_count_gt = 0
        self.bichar_count_gt = 2

        self.token_type = "rnn"

        # embedding settings
        self.char_emb_size = 50
        self.bichar_emb_size = 50
        self.seg_emb_size = 25
        self.dropout = 0.1
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
        self.rnn_cell_type = 'lstm'
        self.rnn_num_layer = 2
        self.rnn_hidden = 256

        # fragment encoder
        self.frag_type = "rnn"
        self.frag_fofe_alpha = 0.5
        self.frag_rnn_cell = "lstm"

        # development config
        self.load_from_cache = False
        self.train_on = True
        self.use_data_set = "full_train"


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
    def __init__(self, d_in, d_hidden):
        super(NonLinearLayer, self).__init__()
        self.fc1 = torch.nn.Linear(d_in, d_hidden)
        self.fc2 = torch.nn.Linear(d_hidden, d_in)

    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x)))
        out += x
        # out = torch.nn.LayerNorm(out)
        return out


class Luban7(torch.nn.Module):
    def __init__(self,
                 char2idx,
                 bichar2idx,
                 seg2idx,
                 label2idx,
                 longest_text_len,
                 ):
        super(Luban7, self).__init__()
        self.char2idx = char2idx
        self.bichar2idx = bichar2idx
        self.seg2idx = seg2idx
        self.label2idx = label2idx

        self.embeds = MixEmbedding(char_vocab_size=len(char2idx),
                                   char_emb_size=config.char_emb_size,
                                   seg_vocab_size=len(seg2idx),
                                   seg_emb_size=config.seg_emb_size,
                                   bichar_vocab_size=len(bichar2idx),
                                   bichar_emb_size=config.bichar_emb_size)
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
                dropout=config.dropout
            )
            token_dim = self.embeds.embedding_dim
        elif config.token_type == "rnn":
            self.token_encoder = BiRNNTokenEncoder(
                cell_type=config.rnn_cell_type,
                num_layers=config.rnn_num_layer,
                input_size=self.embeds.embedding_dim,
                hidden_size=config.rnn_hidden,
                dropout=config.dropout,
            )
            token_dim = config.rnn_hidden
        else:
            token_dim = self.embeds.embedding_dim

        if config.frag_type == "rnn":
            self.fragment_encoder = AllFragEncoder(
                max_span_len=config.max_span_length,
                b2e_encoder=RNNFragEncoder(config.frag_rnn_cell, token_dim, token_dim),
                e2b_encoder=RNNFragEncoder(config.frag_rnn_cell, token_dim, token_dim)
            )
        elif config.frag_type == "fofe":
            self.fragment_encoder = AllFragEncoder(
                max_span_len=config.max_span_length,
                b2e_encoder=FofeFragEncoder(alpha=config.frag_fofe_alpha),
                e2b_encoder=FofeFragEncoder(alpha=config.frag_fofe_alpha)
            )
        elif config.frag_type == "average":
            self.fragment_encoder = AllFragEncoder(
                max_span_len=config.max_span_length,
                b2e_encoder=AverageFragEncoder(),
                e2b_encoder=AverageFragEncoder()
            )
        frag_dim = 2 * token_dim

        self.non_linear = NonLinearLayer(frag_dim, 2 * frag_dim)

        self.label_weight = torch.nn.Parameter(torch.Tensor(frag_dim, len(label2idx)))
        self.label_bias = torch.nn.Parameter(torch.Tensor(len(label2idx)))

        std = 1. / math.sqrt(self.label_weight.size(1))
        self.label_weight.data.uniform_(-std, std)
        if self.label_bias is not None:
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
        labels = list(map(lambda x: x[3], batch_data))
        text_lens = batch_lens(chars)

        pad_chars = batch_pad(chars, self.char2idx['<PAD>'])
        pad_bichars = batch_pad(bichars, self.bichar2idx['<PAD>'])
        pad_segs = batch_pad(segs, self.seg2idx['<PAD>'])

        pad_chars_tensor = torch.tensor(pad_chars).to(device)
        pad_bichars_tensor = torch.tensor(pad_bichars).to(device)
        pad_segs_tensor = torch.tensor(pad_segs).to(device)

        input_embs = self.embeds(pad_chars_tensor,
                                 pad_bichars_tensor,
                                 pad_segs_tensor)

        if config.token_type == 'rnn':
            token_repr = self.token_encoder(input_embs, text_lens)
        elif config.token_type == 'tfer':
            masks = self.token_encoder.gen_masks(pad_chars_tensor)
            token_repr = self.token_encoder(input_embs, masks, text_lens)
        else:
            token_repr = input_embs

        span_reprs = self.fragment_encoder(token_repr, text_lens)

        span_reprs = self.non_linear(span_reprs)

        span_ys = self.gen_span_ys(chars, labels)
        score = F.log_softmax(span_reprs @ self.label_weight + self.label_bias, dim=0)
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
    used_data_set = usable_data_sets[config.use_data_set]
    vocab_folder = "dataset/OntoNotes4/vocab.{}.{}".format(config.char_count_gt, config.bichar_count_gt)
    gen_vocab(data_path=used_data_set[0],
              out_folder=vocab_folder,
              char_count_gt=config.char_count_gt,
              bichar_count_gt=config.bichar_count_gt,
              use_cache=config.load_from_cache)
    char2idx, idx2char = load_vocab("{}/char.vocab".format(vocab_folder))
    bichar2idx, idx2bichar = load_vocab("{}/bichar.vocab".format(vocab_folder))
    seg2idx, idx2seg = load_vocab("{}/seg.vocab".format(vocab_folder))
    label2idx, idx2label = load_vocab("{}/label.vocab".format(vocab_folder))

    idx2str = lambda idx_lst: "".join(map(lambda x: idx2char[x], idx_lst))
    train_set = ConllDataSet(ner_path=used_data_set[0], seg_path=used_data_set[1],
                             char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx, label2idx=label2idx,
                             max_text_len=config.max_sentence_length,
                             max_span_len=config.max_span_length,
                             sort_by_length=True)
    dev_set = ConllDataSet(ner_path=used_data_set[2], seg_path=used_data_set[3],
                           char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx, label2idx=label2idx,
                           sort_by_length=False)
    longest_span_len = max(train_set.longest_span_len, dev_set.longest_span_len)
    longest_text_len = max(train_set.longest_text_len, dev_set.longest_text_len)

    luban7 = Luban7(char2idx=char2idx,
                    bichar2idx=bichar2idx,
                    seg2idx=seg2idx,
                    label2idx=label2idx,
                    longest_text_len=longest_text_len).to(device)
    opt = torch.optim.Adam(luban7.parameters(), lr=0.001)
    # opt = torch.optim.SGD(luban7.parameters(), lr=0.1, momentum=0.9)

    epoch_id = -1
    while True:
        epoch_id += 1
        if epoch_id == 30:
            break
        log("[EPOCH {}]".format(epoch_id))

        """
        Training
        """
        if config.train_on:
            luban7.train()
            train_set.reset(shuffle=True)
            iter_id = 0
            progress = ProgressManager(total=train_set.size)
            log(train_set.size)
            while not train_set.finished:
                iter_id += 1
                batch_data = train_set.next_batch(30)
                batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)

                score, span_ys = luban7(batch_data)

                loss = F.nll_loss(score, torch.tensor(span_ys).to(device))
                # weight=torch.tensor([.1,.3,.3,.3]).to(gpu))
                pred = cast_list(torch.argmax(score, 1).cpu().numpy())
                # print_prf(pred, span_ys, list(idx2label.keys()))
                progress.update(len(batch_data))
                if iter_id % 1 == 0:
                    log("".join([
                        "[{}: {}/{}] ".format(epoch_id, progress.complete_num, train_set.size),
                        "b: {:.4f} / c:{:.4f} / r: {:.4f} "
                            .format(progress.batch_time, progress.cost_time, progress.rest_time),
                        "loss: {:.4f}".format(loss.item()),
                        "accuracy: {:.4f}".format(accuracy(score.detach().cpu().numpy(), span_ys))])
                    )
                    # log(pred)
                    # log(span_ys)
                opt.zero_grad()
                loss.backward()
                opt.step()

        """
        Development
        """
        luban7.eval()
        dev_set.reset(shuffle=False)
        progress = ProgressManager(total=dev_set.size)
        collector = Collector()
        while not dev_set.finished:
            batch_data = dev_set.next_batch(10, fill_batch=True)
            batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)  # type: List[Datum]

            texts = list(map(lambda x: x[0], batch_data))
            text_lens = batch_lens(texts)

            score, span_ys = luban7(batch_data)

            pred = cast_list(torch.argmax(score, 1))

            offset = 0
            for bid in range(len(text_lens)):
                log("[{:>4}] {}".format(
                    progress.complete_num + bid,
                    idx2str(batch_data[bid].chars)))
                enum_spans = enum_span_by_length(text_lens[bid])
                # fragment_score = score[offset: offset + len(enum_spans)]
                # log(fragment_score)
                for sid, span in enumerate(enum_spans):
                    begin_idx, end_idx = span
                    span_offset = sid + offset
                    if pred[span_offset] != 0 or span_ys[span_offset] != 0:
                        log("{:>3}~{:<3}\t{:1}/{:.4f}\t{:>5}/{:<5}\t{}".format(
                            begin_idx, end_idx,
                            pred[span_offset], score[span_offset][pred[span_offset]],
                            idx2label[pred[span_offset]],
                            idx2label[span_ys[span_offset]],
                            idx2str(batch_data[bid].chars[begin_idx: end_idx + 1]),
                        ))
                offset += len(enum_spans)

            progress.update(len(batch_data))


if __name__ == '__main__':
    main()
