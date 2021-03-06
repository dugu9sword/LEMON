from buff import DataSet, create_folder, log, analyze_length_count, analyze_vocab_count, group_fields
from typing import NamedTuple, List, Union, Dict, Tuple
import re
import math
from collections import defaultdict
from functools import lru_cache
from buff import exist_var, load_var, save_var
import os
from program_args import config
import numpy as np

usable_data_sets = {
    "ontogold": ("dataset/ontonotes4/train.mix.bmes",
                 "dataset/ontonotes4/dev.mix.bmes",
                 "dataset/ontonotes4/test.mix.bmes"),
    "ontothu": ("dataset/ontonotes4/train.mix.bmes.thu",
                "dataset/ontonotes4/dev.mix.bmes.thu",
                "dataset/ontonotes4/test.mix.bmes.thu"),
    "weibothu": ("dataset/weibo/train.mix.bmes.thu",
                 "dataset/weibo/dev.mix.bmes.thu",
                 "dataset/weibo/test.mix.bmes.thu"),
    "resumethu": ("dataset/resume/train.mix.bmes.thu",
                  "dataset/resume/dev.mix.bmes.thu",
                  "dataset/resume/test.mix.bmes.thu"),
    "msrathu": ("dataset/msra/train.mix.bmes.thu",
                "dataset/msra/test.mix.bmes.thu",
                "dataset/msra/test.mix.bmes.thu"),
    "msrapred": ("dataset/msra/train.mix.bmes.pred",
                 "dataset/msra/test.mix.bmes.pred",
                 "dataset/msra/test.mix.bmes.pred"),
    "tiny": ("dataset/ontonotes4/tiny.mix.bmes",
             "dataset/ontonotes4/tiny.mix.bmes",
             "dataset/ontonotes4/tiny.mix.bmes")}


def load_sentences(file_path, sep=r"\s+"):
    ret = []
    sentence = []
    for line in open(file_path, encoding='utf8'):
        line = line.strip("\n")
        if line == "":
            if not sentence == []:
                ret.append(sentence)
                sentence = []
        else:
            sentence.append(re.split(sep, line))
    return ret


def pair(a, b):
    return a, b


FragIdx = NamedTuple("FragIdx", [("bid", int), ("eid", int)])

SpanLabel = NamedTuple("SpanLabel", [("b", int),
                                     ("e", int),
                                     ("y", int)])
SpanPred = NamedTuple("SpanPred", [("b", int),
                                   ("e", int),
                                   ("pred", int),
                                   ("prob", float)])
Datum = NamedTuple("Datum", [("chars", List[int]),
                             ("bichars", List[int]),
                             ("segs", List[int]),
                             ("poss", List[int]),
                             ("ners", List[int]),
                             ("lexmatches", object),
                             ("labels", List[SpanLabel])])


class Sp:
    pad = "<pad>"
    oov = "<oov>"
    sos = "<sos>"
    eos = "<eos>"
    non = "<non>"


def load_vocab(vocab_path):
    token2idx = {}
    for line in open(vocab_path, encoding='utf8'):
        split = line.split(" ")
        token2idx[split[0]] = int(split[1])
    idx2token = {v: k for k, v in token2idx.items()}
    return token2idx, idx2token


# gen_vocab("dataset/ontonotes4/train.mix.bmes", out_folder="dataset/ontonotes4/vocab")

def gen_vocab(data_path, out_folder,
              char_count_gt=2,
              bichar_count_gt=2,
              ignore_tag_bmes=False,
              use_cache=False):
    if use_cache and os.path.exists(out_folder):
        log("cache for vocab exists.")
        return
    sentences = load_sentences(data_path)

    char_count = defaultdict(lambda: 0)
    bichar_count = defaultdict(lambda: 0)
    ner_labels = []  # BE-*
    pos_vocab = {Sp.pad: 0}
    for sentence in sentences:
        for line_idx, line in enumerate(sentence):
            char_count[line[0]] += 1
            if line[3] not in ner_labels:
                ner_labels.append(line[3])
            pos = line[2]
            if ignore_tag_bmes:
                pos = pos[2:]
            if pos not in pos_vocab:
                pos_vocab[pos] = len(pos_vocab)
            if line_idx < len(sentence) - 1:
                bichar_count[line[0] + sentence[line_idx + 1][0]] += 1
            else:
                bichar_count[line[0] + Sp.eos] += 1

    char_count = dict(sorted(char_count.items(), key=lambda x: x[1], reverse=True))
    bichar_count = dict(sorted(bichar_count.items(), key=lambda x: x[1], reverse=True))

    # gen char vocab
    char_vocab = {Sp.pad: 0, Sp.oov: 1, Sp.sos: 2, Sp.eos: 3}
    for i, k in enumerate(char_count.keys()):
        if char_count[k] > char_count_gt:
            char_vocab[k] = len(char_vocab)
    analyze_vocab_count(char_count)

    # gen char vocab
    bichar_vocab = {Sp.pad: 0, Sp.oov: 1}
    for i, k in enumerate(bichar_count.keys()):
        if bichar_count[k] > bichar_count_gt:
            bichar_vocab[k] = len(bichar_vocab)
    analyze_vocab_count(bichar_count)

    # seg vocab
    seg_vocab = {Sp.pad: 0, "B": 1, "M": 2, "E": 3, "S": 4}

    # ner vocab / BMES mode
    ner_vocab = {Sp.pad: 0, Sp.sos: 1, Sp.eos: 2}
    for tag in ner_labels:
        ner_vocab[tag] = len(ner_vocab)

    # gen label vocab
    label_vocab = {"NONE": 0}
    for label in ner_labels:
        found = re.search(".*-(.*)", label)
        if found:
            if found.group(1) not in label_vocab:
                label_vocab[found.group(1)] = len(label_vocab)

    # write to file
    create_folder(out_folder)
    for ele in {"char.vocab": char_vocab,
                "bichar.vocab": bichar_vocab,
                "seg.vocab": seg_vocab,
                "pos.vocab": pos_vocab,
                "ner.vocab": ner_vocab,
                "label.vocab": label_vocab,
                }.items():
        f_out = open("{}/{}".format(out_folder, ele[0]), "w", encoding='utf8')
        for k, v in ele[1].items():
            f_out.write("{} {}\n".format(k, v))
        f_out.close()


""" Match dict """
max_match_num = config.max_match_num
match2idx_naive = {"full_match": 0,
                   "prefix_match": 1,
                   "suffix_match": 2,
                   "inter_match": 3,
                   "no_match": 4}
idx2match_naive = {v: k for k, v in match2idx_naive.items()}

match2idx_middle = {
    "full_match": 0,
    "pre_1": 1,
    "suff_1": 2,
    "pre_2": 3,
    "suff_2": 4,
    "inter_match": 5,
    "no_match": 6
}
idx2match_middle = {v: k for k, v in match2idx_middle.items()}

match2idx_mix = {
    "full_match": 0,
    "pre_1": 1,
    "pre_2": 2,
    "suff_1": 3,
    "suff_2": 4,
    "pre_3": 5,
    "suff_3": 6,
    "inter_match": 7,
    "no_match": 8
}
idx2match_mix = {v: k for k, v in match2idx_mix.items()}


class ConllDataSet(DataSet):

    def __init__(self, data_path,
                 lexicon2idx,
                 char2idx, bichar2idx, seg2idx, pos2idx, ner2idx, label2idx,
                 ignore_pos_bmes=False,
                 max_text_len=19260817,
                 max_span_len=19260817,
                 sort_by_length=False):
        super(ConllDataSet, self).__init__()
        if config.lexicon_emb_pretrain != "off":
            self.word2idx = lexicon2idx
            self.idx2word = {v: k for k, v in self.word2idx.items()}

        sentences = load_sentences(data_path)

        self.__max_text_len = max_text_len
        self.__max_span_len = max_span_len
        self.__longest_text_len = -1
        self.__longest_span_len = -1

        __span_length_count = defaultdict(lambda: 0)
        __sentence_length_count = defaultdict(lambda: 0)

        for sid in range(len(sentences)):
            chars, bichars, segs, labels, poss, ners = [], [], [], [], [], []

            sen = sentences[sid]
            sen_len = len(sen)

            for cid in range(sen_len):
                char = sen[cid][0]
                chars.append(char2idx[char] if char in char2idx else char2idx[Sp.oov])

                bichar = char + sen[cid + 1][0] if cid < sen_len - 1 else char + Sp.eos
                bichars.append(bichar2idx[bichar] if bichar in bichar2idx else bichar2idx[Sp.oov])

                segs.append(seg2idx[sen[cid][1]])

                pos = sen[cid][2]
                if ignore_pos_bmes:
                    pos = pos[2:]
                poss.append(pos2idx[pos])

                ners.append(ner2idx[sen[cid][3]])

                if re.match(r"^[BS]", sen[cid][3]):
                    state, label = sen[cid][3].split("-")
                    label_b = cid
                    label_e = cid
                    label_y = label2idx[label]
                    if state == 'B':
                        while True:
                            next_state, _ = sen[label_e][3].split("-")
                            if next_state == "E":
                                break
                            label_e += 1
                    if state == 'S':
                        pass

                    __span_length_count[label_e - label_b + 1] += 1
                    if label_e - label_b + 1 <= max_span_len:
                        labels.append(SpanLabel(b=label_b, e=label_e, y=label_y))
                        self.__longest_span_len = max(self.__longest_span_len, label_e - label_b + 1)

            __sentence_length_count[len(chars)] += 1
            if len(chars) < max_text_len:
                if config.lexicon_emb_pretrain != "off":
                    if config.match_mode == "naive":
                        lexmatches = match_lex_naive(group_fields(sen, indices=0),
                                                     lexicon2idx=lexicon2idx)
                        # for ele in lexmatches:
                        #     print("".join(group_fields(sen, indices=0)[ele[0][0]: ele[0][1]+ 1]))
                        #     for word_idx, match_type in ele[1]:
                        #         print(">>\t" ,self.idx2word[word_idx], idx2match_naive[match_type])
                    elif config.match_mode == "middle":
                        lexmatches = match_lex_middle(group_fields(sen, indices=0),
                                                      lexicon2idx=lexicon2idx)
                        # for ele in lexmatches:
                        #     print("".join(group_fields(sen, indices=0)[ele[0][0]: ele[0][1] + 1]))
                        #     for word_idx, match_type in ele[1]:
                        #         print(">>\t", self.idx2word[word_idx], idx2match_middle[match_type])
                    elif config.match_mode == "mix":
                        lexmatches = match_lex_mix(group_fields(sen, indices=0),
                                                   lexicon2idx=lexicon2idx)
                        # for ele in lexmatches:
                        #     print("".join(group_fields(sen, indices=0)[ele[0][0]: ele[0][1] + 1]))
                        #     for word_idx, match_type in ele[1]:
                        #         print(">>\t", self.idx2word[word_idx], idx2match_mix[match_type])
                    elif config.match_mode == "off":
                        lexmatches = None
                    else:
                        raise Exception
                else:
                    lexmatches = None

                self.data.append(Datum(chars=chars, bichars=bichars, segs=segs,
                                       poss=poss, ners=ners, labels=labels,
                                       lexmatches=lexmatches))
                self.__longest_text_len = max(self.__longest_text_len, len(chars))

        if sort_by_length:
            self.data = sorted(self.data, key=lambda x: len(x[0]), reverse=True)
        log("Dataset statistics for {}".format(data_path))
        log("Sentence")
        analyze_length_count(__sentence_length_count)
        log("Span")
        analyze_length_count(__span_length_count)

    @property
    def longest_text_len(self):
        return self.__longest_text_len

    @property
    def longest_span_len(self):
        return self.__longest_span_len


@lru_cache(maxsize=None)
def fragments(sentence_len, max_span_len) -> List[FragIdx]:
    ret = []
    for i in range(sentence_len):
        for j in range(i, i + max_span_len):
            if j == sentence_len:
                break
            ret.append(FragIdx(i, j))
    return ret


def match_lex_mix(chars, lexicon2idx):
    mapping_dict = {}  # type: Dict[FragIdx, int]
    ret = []
    length = len(chars)
    # print(self.__max_span_len)
    # print(fragments(length, self.__max_span_len))
    for i, j in fragments(length, config.max_span_length):
        lexicon = "".join(chars[i:j + 1])
        if lexicon in lexicon2idx:
            mapping_dict[FragIdx(i, j)] = lexicon2idx[lexicon]
    for i, j in fragments(length, config.max_span_length):
        matched_lexicons = []
        for sub_i in upto(i, j):
            for sub_j in upto(sub_i, j):
                if FragIdx(sub_i, sub_j) in mapping_dict:
                    if sub_i == i and sub_j == j:
                        m_idx = match2idx_mix['full_match']
                    elif sub_i == i and sub_j != j:
                        if sub_j == i:
                            m_idx = match2idx_mix["pre_1"]
                        elif sub_j == i + 1:
                            m_idx = match2idx_mix["pre_2"]
                        else:
                            m_idx = match2idx_mix["pre_3"]
                    elif sub_i != i and sub_j == j:
                        if j == sub_i:
                            m_idx = match2idx_mix["suff_1"]
                        elif j == sub_i + 1:
                            m_idx = match2idx_mix["suff_2"]
                        else:
                            m_idx = match2idx_mix["suff_3"]
                    elif sub_i != sub_j:
                        m_idx = match2idx_mix["inter_match"]
                    else:
                        continue
                    matched_lexicons.append(
                        pair(mapping_dict[FragIdx(sub_i, sub_j)], m_idx)
                    )
        matched_lexicons.sort(key=lambda x: x[1])
        matched_lexicons = matched_lexicons[:max_match_num]
        if len(matched_lexicons) == 0:
            matched_lexicons.append(
                pair(lexicon2idx[Sp.non], match2idx_mix["no_match"])
            )
        ret.append(
            (pair(i, j), matched_lexicons)
        )
    return ret


def upto(a, b) -> range:
    return range(a, b + 1)


def downto(b, a) -> range:
    return range(b, a - 1, -1)


def match_lex_middle(chars, lexicon2idx):
    mapping_dict = {}  # type: Dict[FragIdx, int]
    ret = []
    length = len(chars)
    # print(self.__max_span_len)
    # print(fragments(length, self.__max_span_len))
    for i, j in fragments(length, config.max_span_length):
        lexicon = "".join(chars[i:j + 1])
        if lexicon in lexicon2idx:
            mapping_dict[FragIdx(i, j)] = lexicon2idx[lexicon]
    for i, j in fragments(length, config.max_span_length):
        matched_lexicons = []
        for sub_i in upto(i, j):
            for sub_j in upto(sub_i, j):
                if FragIdx(sub_i, sub_j) in mapping_dict:
                    if sub_i == i and sub_j == j:
                        m_idx = match2idx_middle['full_match']
                    elif sub_i == i and sub_j != j:
                        if sub_j == i:
                            m_idx = match2idx_middle["pre_1"]
                        else:
                            m_idx = match2idx_middle["pre_2"]
                    elif sub_i != i and sub_j == j:
                        if j == sub_i:
                            m_idx = match2idx_middle["suff_1"]
                        else:
                            m_idx = match2idx_middle["suff_2"]
                    elif sub_i != sub_j:
                        m_idx = match2idx_middle["inter_match"]
                    else:
                        continue
                    matched_lexicons.append(
                        pair(mapping_dict[FragIdx(sub_i, sub_j)], m_idx)
                    )
        matched_lexicons.sort(key=lambda x: x[1])
        matched_lexicons = matched_lexicons[:max_match_num]
        if len(matched_lexicons) == 0:
            matched_lexicons.append(
                pair(lexicon2idx[Sp.non], match2idx_middle["no_match"])
            )
        ret.append(
            (pair(i, j), matched_lexicons)
        )
    return ret


def match_lex_naive(chars, lexicon2idx):
    mapping_dict = {}  # type: Dict[FragIdx, int]
    ret = []
    length = len(chars)
    # print(self.__max_span_len)
    # print(fragments(length, self.__max_span_len))
    for i, j in fragments(length, config.max_span_length):
        if i != j:
            lexicon = "".join(chars[i:j + 1])
            if lexicon in lexicon2idx:
                mapping_dict[FragIdx(i, j)] = lexicon2idx[lexicon]
    for i, j in fragments(length, config.max_span_length):
        matched_lexicons = []
        for sub_i in upto(i, j):
            for sub_j in upto(sub_i, j):
                if FragIdx(sub_i, sub_j) in mapping_dict:
                    if sub_i == i:
                        if sub_j == j:
                            m_idx = match2idx_naive["full_match"]
                        else:
                            m_idx = match2idx_naive['prefix_match']
                    else:
                        if sub_j == j:
                            m_idx = match2idx_naive['suffix_match']
                        else:
                            m_idx = match2idx_naive['inter_match']
                    matched_lexicons.append(
                        pair(mapping_dict[FragIdx(sub_i, sub_j)], m_idx)
                    )
        matched_lexicons.sort(key=lambda x: x[1])
        matched_lexicons = matched_lexicons[:max_match_num]
        if len(matched_lexicons) == 0:
            matched_lexicons.append(
                pair(lexicon2idx[Sp.non], match2idx_naive["no_match"])
            )
        ret.append(
            (pair(i, j), matched_lexicons)
        )
    return ret


def gen_lexicon_vocab(*data_paths, word2vec_path, out_folder, use_cache=False):
    if use_cache and os.path.exists("{}/lexicon.vocab".format(out_folder)):
        log("cache for lexicon vocab exists.")
        return
    words = set()
    for line in open(word2vec_path, encoding="utf8", errors="ignore"):
        word = re.split(r"\s+", line.strip())[0]
        words.add(word)

    lexicon = {Sp.pad: 0, Sp.oov: 1, Sp.non: 2, Sp.sos: 3, Sp.eos: 4}
    for data_path in data_paths:
        print("Gen lexicon for", data_path)
        sentences = load_sentences(data_path)
        for sid, sentence in enumerate(sentences):
            chars = group_fields(sentence, indices=0)
            for i, j in fragments(len(chars), config.max_span_length):
                frag = "".join(chars[i:j + 1])
                if frag not in lexicon and frag in words:
                    lexicon[frag] = len(lexicon)
    create_folder(out_folder)
    f_out = open("{}/lexicon.vocab".format(out_folder), "w", encoding='utf8')
    for k, v in lexicon.items():
        f_out.write("{} {}\n".format(k, v))
    f_out.close()
