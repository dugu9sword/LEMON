# -*- coding: UTF-8 -*-

from buff import *
from typing import NamedTuple
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import os
import math
from collections import defaultdict


class ProgramArgs(argparse.Namespace):
    def __init__(self):
        super(ProgramArgs, self).__init__()
        self.max_span_length = 10
        self.max_sentence_length = 120


parser = argparse.ArgumentParser()
nsp = ProgramArgs()
for key, value in nsp.__dict__.items():
    parser.add_argument('-{}'.format(key),
                        action='store',
                        default=value,
                        type=type(value),
                        dest=str(key))
config = parser.parse_args(namespace=nsp)  # type: ProgramArgs

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
if torch.cuda.is_available():
    dev = torch.device("cuda:0")
else:
    dev = torch.device("cpu")

SpanLabel = NamedTuple("SpanLabel", [("b", int),
                                     ("e", int),
                                     ("y", int)])
SpanPred = NamedTuple("SpanPred", [("b", int),
                                   ("e", int),
                                   ("pred", int),
                                   ("prob", float)])
Datum = NamedTuple("Datum", [("texts", List[int]),
                             ("labels", List[SpanLabel])])


class SeqLabelDataSet(DataSet):
    @classmethod
    def gen_vocab(cls, path, out_folder):
        text_vocab = {"<PAD>": 0, "<OOV>": 1}
        label_vocab = {"NONE": 0}
        _labels = []  # BE-*
        for line in open(path, encoding='utf8'):
            line = line.strip("\n")
            if line != "":
                pm = re.search("([^\s]*)\s(.*)", line)
                text = pm.group(1)
                label = pm.group(2)
                if text not in text_vocab:
                    text_vocab[text] = len(text_vocab)
                if label not in _labels:
                    _labels.append(label)
        for label in _labels:
            pm = re.search(".*-(.*)", label)
            if pm:
                if pm.group(1) not in label_vocab:
                    label_vocab[pm.group(1)] = len(label_vocab)

        create_folder(out_folder)
        f_out_text = open("{}/text.vocab".format(out_folder), "w", encoding='utf8')
        for k, v in text_vocab.items():
            f_out_text.write("{} {}\n".format(k, v))
        f_out_text.close()
        f_out_label = open("{}/label.vocab".format(out_folder), "w", encoding='utf8')
        for k, v in label_vocab.items():
            f_out_label.write("{} {}\n".format(k, v))
        f_out_label.close()

    @classmethod
    def load_vocab(cls, path):
        text2idx = {}
        label2idx = {}
        for line in open("{}/text.vocab".format(path), encoding='utf8'):
            split = line.split(" ")
            text2idx[split[0]] = int(split[1])
        for line in open("{}/label.vocab".format(path), encoding='utf8'):
            split = line.split(" ")
            label2idx[split[0]] = int(split[1])
        idx2text = {v: k for k, v in text2idx.items()}
        idx2label = {v: k for k, v in label2idx.items()}
        return text2idx, label2idx, idx2text, idx2label

    def __init__(self, path, text2idx, label2idx,
                 max_text_len=math.inf,
                 max_span_len=math.inf):
        super(SeqLabelDataSet, self).__init__()

        __span_length_count = defaultdict(lambda: 0)
        __sentence_length_count = defaultdict(lambda: 0)
        texts, labels = [], []
        _b, _e, _y = -1, -1, -1
        inner_id = -1
        for line in open(path, encoding='utf8'):
            line = line.strip("\n")
            if line == "":
                if len(texts) > 0:
                    __sentence_length_count[len(texts)] += 1
                    if len(texts) < max_text_len:
                        self.data.append(Datum(texts=texts, labels=labels))
                texts, labels = [], []
                _b, _e, _y = -1, -1, -1
                inner_id = -1
                continue

            inner_id += 1
            split = line.split(" ")

            if split[0] in text2idx:
                texts.append(text2idx[split[0]])
            else:
                texts.append(text2idx['<OOV>'])
            if split[1] != 'O':
                state, label = split[1].split("-")
                if state == 'B':
                    _b = inner_id
                elif state == 'M':
                    continue
                elif state == 'E':
                    _e = inner_id
                elif state == 'S':
                    _b = inner_id
                    _e = inner_id
                if _e != -1:
                    _y = label2idx[label]
                if _y != -1:
                    __span_length_count[_e - _b + 1] += 1
                    if _e - _b + 1 <= max_span_len:
                        labels.append(SpanLabel(b=_b, e=_e, y=_y))
                    _b, _e, _y = -1, -1, -1
        # Maybe the last sentence is not appended due to no more blank lines
        if len(texts) > 0:
            __sentence_length_count[len(texts)] += 1
            if len(texts) < max_text_len:
                self.data.append(Datum(texts=texts, labels=labels))

        analyze_length_count(__sentence_length_count)
        analyze_length_count(__span_length_count)


class Luban7(torch.nn.Module):
    def __init__(self,
                 text2idx,
                 label2idx,
                 emb_size,
                 hidden_size,
                 num_layers,
                 ):
        super(Luban7, self).__init__()
        self.text2idx = text2idx
        self.label2idx = label2idx
        self.embedding = torch.nn.Embedding(num_embeddings=len(text2idx),
                                            embedding_dim=emb_size)
        self.rnn = torch.nn.LSTM(input_size=emb_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=True,
                                 batch_first=True,
                                 # dropout=0.2)
                                 )

        self.non_linear = torch.nn.Sequential(
            torch.nn.Linear(2 * hidden_size, hidden_size),
            torch.nn.Dropout(),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size),
        )

        self.label_weight = torch.nn.Parameter(torch.Tensor(hidden_size, len(label2idx)))
        self.label_bias = torch.nn.Parameter(torch.Tensor(len(label2idx)))

        std = 1. / math.sqrt(self.label_weight.size(1))
        self.label_weight.data.uniform_(-std, std)
        if self.label_bias is not None:
            self.label_bias.data.uniform_(-std, std)

    def forward(self, batch_data):
        texts = list(map(lambda x: x[0], batch_data))
        labels = list(map(lambda x: x[1], batch_data))
        text_lens = batch_lens(texts)

        pad_texts = batch_pad(texts, self.text2idx['<PAD>'])
        input_embs = self.embedding(torch.tensor(pad_texts).to(dev))
        packed_input_embs = pack_padded_sequence(input_embs, text_lens, batch_first=True)
        rnn_out, _ = self.rnn(packed_input_embs)
        rnn_out, _ = pad_packed_sequence(rnn_out, batch_first=True)
        token_repr = self.non_linear(rnn_out)

        span_reprs = []
        span_ys = []
        for bid in range(len(batch_data)):
            for begin_token_idx in range(text_lens[bid]):
                for span_len in range(1, config.max_span_length + 1):
                    end_token_idx = begin_token_idx + span_len - 1
                    if end_token_idx < text_lens[bid]:
                        span_repr = torch.mean(token_repr[bid][begin_token_idx:end_token_idx + 1], 0)
                        span_reprs.append(span_repr)
                        has_label = False
                        for span in labels[bid]:
                            if span.b == begin_token_idx and span.e == end_token_idx:
                                span_ys.append(span.y)
                                print(Color.blue("{}, {}: {}".format(span.b,
                                                                     span.e,
                                                                     span.y)))
                                has_label = True
                                break
                        if not has_label:
                            span_ys.append(0)
                        # print(begin_token_idx, ", ", end_token_idx, " of ", text_lens[bid])
        span_reprs = torch.stack(span_reprs)
        score = F.log_softmax(span_reprs @ self.label_weight + self.label_bias, dim=0)

        return score, span_ys


def span_num(sentence_length):
    return (sentence_length + sentence_length - config.max_span_length + 1) * config.max_span_length / 2


def main():
    SeqLabelDataSet.gen_vocab("dataset/OntoNotes4/small_train.char.bmes", "OntoNotes4/vocab")
    text2idx, label2idx, idx2text, idx2label = SeqLabelDataSet.load_vocab("OntoNotes4/vocab")
    train_set = SeqLabelDataSet("dataset/OntoNotes4/small_train.char.bmes", text2idx, label2idx,
                                config.max_sentence_length,
                                config.max_span_length)
    dev_set = SeqLabelDataSet("dataset/OntoNotes4/small_train.char.bmes", text2idx, label2idx)
    luban7 = Luban7(text2idx=text2idx,
                    label2idx=label2idx,
                    emb_size=50,
                    hidden_size=128,
                    num_layers=2).to(dev)
    opt = torch.optim.Adam(luban7.parameters(), lr=0.001)
    # opt = torch.optim.SGD(luban7.parameters(), lr=0.1, momentum=0.9)

    epoch_id = -1
    while True:
        epoch_id += 1

        train_on = True
        if train_on:
            luban7.train()
            train_set.reset()
            iter_id = 0
            progress = ProgressManager(total=train_set.size)
            print(train_set.size)
            while not train_set.finished:
                iter_id += 1
                batch_data = train_set.next_batch(30)
                batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)

                score, span_ys = luban7(batch_data)

                loss = F.nll_loss(score, torch.tensor(span_ys).to(dev))
                pred = cast_list(torch.argmax(score, 1).cpu().numpy())
                # print_prf(pred, span_ys, list(idx2label.keys()))
                progress.update(len(batch_data))
                if iter_id % 1 == 0:
                    print(
                        "[{}: {}/{}] ".format(epoch_id, progress.complete_num, train_set.size),
                        "b: {:.4f} / c:{:.4f} / r: {:.4f} "
                            .format(progress.batch_time, progress.cost_time, progress.rest_time),
                        "loss: {:.4f}".format(loss.item()),
                        "accuracy: {:.4f}".format(accuracy(score.detach().cpu().numpy(), span_ys))
                    )
                opt.zero_grad()
                loss.backward()
                opt.step()

        luban7.eval()
        dev_set.reset(shuffle=False)
        collector = Collector()
        while not dev_set.finished:
            batch_data = dev_set.next_batch(2, fill_batch=True)
            batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True) # type: List[Datum]

            texts = list(map(lambda x: x[0], batch_data))
            labels = list(map(lambda x: x[1], batch_data))
            text_lens = batch_lens(texts)

            score, span_ys = luban7(batch_data)

            pred = cast_list(torch.argmax(score, 1))
            pointer = 0
            for bid in range(len(batch_data)):
                print("".join(map(lambda x: idx2text[x], batch_data[bid].texts)))
                span_offset = 0
                for begin_token_idx in range(text_lens[bid]):
                    for span_len in range(1, config.max_span_length + 1):
                        end_token_idx = begin_token_idx + span_len - 1
                        if end_token_idx < text_lens[bid]:
                            offset_pointer = pointer + span_offset
                            if pred[offset_pointer] != 0:
                                print("{} - {} : {} / {:.4f} # {} : {} $ [{} √]".format(
                                    begin_token_idx,
                                    end_token_idx,
                                    pred[offset_pointer],
                                    score[offset_pointer][pred[offset_pointer]],
                                    "".join(map(lambda x: idx2text[x], batch_data[bid].texts[begin_token_idx: end_token_idx + 1])),
                                    idx2label[pred[offset_pointer]],
                                    idx2label[span_ys[offset_pointer]]
                                ))
                            span_offset += 1
                assert span_offset == span_num(text_lens[bid])
                pointer += span_offset
            # collector.collect(pred, span_ys)
        # pred, span_ys = collector.collected()
        # print(pred)
        # print(span_ys)
        # print_prf(pred, span_ys, list(idx2label.keys()))
        print("accuracy: {:.4f}".format(sum(np.array(pred) == np.array(span_ys)) / len(pred)))


if __name__ == '__main__':
    main()
