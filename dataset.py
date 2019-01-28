from buff import DataSet, create_folder, log, analyze_length_count, analyze_vocab_count, group_fields
from typing import NamedTuple, List, Union, Dict, Tuple
import re
import math
from collections import defaultdict
from functools import lru_cache
import os

usable_data_sets = {"full": ("dataset/ontonotes4/train.mix.bmes",
                             "dataset/ontonotes4/dev.mix.bmes",
                             "dataset/ontonotes4/test.mix.bmes"),
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


FragIdx = NamedTuple("FragIdx", [("bid", int), ("eid", int)])
MatchedWord = NamedTuple("MatchedWord", [("word_idx", int), ("match_type", int)])
FragMatches = NamedTuple("FragMatches", [("frag", FragIdx), ("matches", List[MatchedWord])])
AllFragMatches = List[FragMatches]

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
                             ("matches", AllFragMatches),
                             ("labels", List[SpanLabel])])


class Sp:
    pad = "<pad>"
    oov = "<oov>"
    sos = "<sos>"
    eos = "<eos>"


def load_vocab(vocab_path):
    token2idx = {}
    for line in open(vocab_path, encoding='utf8'):
        split = line.split(" ")
        token2idx[split[0]] = int(split[1])
    idx2token = {v: k for k, v in token2idx.items()}
    return token2idx, idx2token


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


# gen_vocab("dataset/ontonotes4/train.mix.bmes", out_folder="dataset/ontonotes4/vocab")
match2idx = {"full_match": 0,
             "prefix_match": 1,
             "suffix_match": 2,
             "inter_match": 3}
idx2match = {v: k for k, v in match2idx.items()}


class ConllDataSet(DataSet):

    def __init__(self, data_path,
                 word2idx,
                 char2idx, bichar2idx, seg2idx, pos2idx, ner2idx, label2idx,
                 ignore_pos_bmes=False,
                 max_text_len=19260814,
                 max_span_len=19260814,
                 max_match_num=8,
                 sort_by_length=False):
        super(ConllDataSet, self).__init__()
        self.word2idx = word2idx
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
                __sentence_length_count[len(chars)] += 1

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

            if len(chars) < max_text_len:
                all_frag_matches = match_word_dict(group_fields(sen, indices=0),
                                                   word2idx=word2idx,
                                                   max_span_len=max_span_len,
                                                   max_match_num=max_match_num)
                # for ele in all_frag_matches:
                #     print("".join(group_fields(sen, indices=0)[ele.frag.bid: ele.frag.eid + 1]))
                #     for word_idx, match_type in ele.matches:
                #         print(self.idx2word[word_idx])
                self.data.append(Datum(chars=chars, bichars=bichars, segs=segs,
                                       poss=poss, ners=ners, labels=labels,
                                       matches=all_frag_matches))
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
        for j in range(i + 1, i + max_span_len + 1):
            if j >= sentence_len:
                break
            ret.append(FragIdx(i, j - 1))
    return ret


def match_word_dict(chars, word2idx, max_span_len, max_match_num) -> AllFragMatches:
    mapping_dict = {}  # type: Dict[FragIdx, int]
    ret = []  # type: AllFragMatches
    length = len(chars)
    # print(self.__max_span_len)
    # print(fragments(length, self.__max_span_len))
    for i, j in fragments(length, max_span_len):
        if i != j:
            word = "".join(chars[i:j + 1])
            if word in word2idx:
                mapping_dict[FragIdx(i, j)] = word2idx[word]
    for i, j in fragments(length, max_span_len):
        matched_words = []
        for sub_i in range(i, j + 1):
            for sub_j in range(sub_i, j + 1):
                if FragIdx(sub_i, sub_j) in mapping_dict:
                    if sub_i == i:
                        if sub_j == j:
                            m_idx = match2idx['full_match']
                        else:
                            m_idx = match2idx['prefix_match']
                    else:
                        if sub_j == j:
                            m_idx = match2idx['suffix_match']
                        else:
                            m_idx = match2idx['inter_match']
                    matched_words.append(MatchedWord(word_idx=mapping_dict[FragIdx(sub_i, sub_j)],
                                                     match_type=m_idx))
                    matched_words.sort(key=lambda x: x[1])
                    matched_words = matched_words[:max_match_num]
        ret.append(FragMatches(FragIdx(i, j), matched_words))
    return ret
