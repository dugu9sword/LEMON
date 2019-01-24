from typing import List
import numpy as np
import random
import copy


def count_token(file_path, verbose=False):
    file = open(file_path, encoding="utf8")
    _count = {}
    while True:
        line = file.readline()
        if line == '':
            break
        arr = line[:-1]
        if arr is not None:
            arr = arr.split(' ')
        for ele in arr:
            if ele in _count:
                _count[ele] += 1
            else:
                _count[ele] = 1
    if verbose:
        total = sum(map(lambda item: item[1], _count.items()))
        print("Total count: {}".format(total))
        num_covered = 0
        stones = [0.9, 0.95, 0.96, 0.97, 0.98, 0.99]
        stone = stones.pop(0)
        k = 0
        for ele in sorted(_count.items(), key=lambda x: x[1], reverse=True):
            k += 1
            num_covered += ele[1]
            if num_covered / total > stone:
                print("Top {:6} covers {}".format(k, stone))
                if len(stones) == 0:
                    break
                stone = stones.pop(0)
    return _count


def merge_count(*counts):
    tmp_count = {}
    for count in counts:
        for ele in count:
            if ele in tmp_count:
                tmp_count[ele] = tmp_count[ele] + count[ele]
            else:
                tmp_count[ele] = count[ele]
    return tmp_count


def build_vocab(count, topk=None):
    assert topk is not None
    _vocab = {
        Vocab.nil_token: Vocab.nil_id,
        Vocab.eos_token: Vocab.eos_id,
        Vocab.unk_token: Vocab.unk_id
    }
    for ele in sorted(count.items(), key=lambda x: x[1], reverse=True):
        _vocab[ele[0]] = len(_vocab)
        if len(_vocab) > topk:
            break
    _rev_vocab = {item[1]: item[0] for item in _vocab.items()}
    return Vocab(_vocab, _rev_vocab)


class Vocab:
    eos_token, eos_id = "<EOS>", 0
    unk_token, unk_id = "<UNK>", 1
    nil_token, nil_id = "<NIL>", 2

    def __init__(self, w2i_dct, i2w_dct):
        self.__w2i_dct = w2i_dct
        self.__i2w_dct = i2w_dct

    def seq2idx(self, seq) -> list:
        return list(map(lambda x: self.__w2i_dct[x] if x in self.__w2i_dct else self.unk_id, seq))

    def idx2seq(self, idx: list) -> list:
        if self.eos_id in idx:
            idx = idx[:idx.index(self.eos_id)]
        return list(map(lambda x: self.__i2w_dct[x], idx))

    def perplexity(self, idx: list, log_prob: list) -> float:
        if self.eos_id in idx:
            log_prob = log_prob[:idx.index(self.eos_id)]
        print(idx)
        N = len(log_prob)
        return np.exp(-1 / (N + 0.001) * np.sum(log_prob))

    def word2idx(self, word):
        return self.__w2i_dct[word]

    def idx2word(self, idx):
        return self.__i2w_dct[idx]

    def file2index(self, path) -> List:
        file = open(path, encoding="utf8")
        ret = []
        process = 0
        while True:
            raw_post = file.readline()
            if raw_post == '':
                break
            arr = raw_post[:-1]
            arr = arr.split(' ')
            ret.append(self.seq2idx(arr))
            process += 1
            print("Process sentence {} in {}".format(process, path))
        return ret

    @property
    def size(self):
        return len(self.__w2i_dct)

    @property
    def w2i_dct(self):
        return self.__w2i_dct

    @property
    def i2w_dct(self):
        return self.__i2w_dct


def lst2str(lst: list) -> str:
    return ''.join(lst)


def random_drop(idx: List, drop_rate) -> List:
    assert 0.0 < drop_rate < 0.5
    ret = list(filter(lambda x: x is not None,
                      map(lambda x: None if random.random() < drop_rate else x, idx)))
    if len(ret) == 0:
        return ret
    return ret


def __shuffle_slice(lst: List, start: int, stop: int):
    cp_lst = copy.copy(lst)
    # Fisher Yates Shuffle
    for i in range(start, stop):
        idx = random.randrange(i, stop)
        cp_lst[i], cp_lst[idx] = cp_lst[idx], cp_lst[i]
        i += 1
    return cp_lst


def random_shuffle_slice(lst: List, width: int) -> List:
    start = random.randrange(0, len(lst))
    stop = min(start + width, len(lst))
    return __shuffle_slice(lst, start, stop)


def batch_random_shuffle_slice(idx: List[List], width: int) -> List[List]:
    return list(map(lambda x: random_shuffle_slice(x, width), idx))


def batch_drop(idx: List[List], drop_rate) -> List[List]:
    return list(map(lambda x: random_drop(x, drop_rate), idx))


def batch_pad(idx: List[List], pad_ele=0) -> List[List]:
    pad_len = max(map(len, idx))
    return list(map(lambda x: x + [pad_ele] * (pad_len - len(x)), idx))


def batch_mask(idx: List[List], mask_zero=True) -> List[List]:
    if mask_zero:
        good_ele, mask_ele = 1, 0
    else:
        good_ele, mask_ele = 0, 1
    max_len = max(map(len, idx))
    return list(map(lambda x: [good_ele] * len(x) + [mask_ele] * (max_len - len(x)), idx))


def batch_mask_by_len(lens: List[int], mask_zero=True) -> List[List]:
    if mask_zero:
        good_ele, mask_ele = 1, 0
    else:
        good_ele, mask_ele = 0, 1
    max_len = max(lens)
    return list(map(lambda x: [good_ele] * x + [mask_ele] * (max_len - x), lens))


def batch_append(idx: List[List], append_ele, before=False) -> List[List]:
    if not before:
        return list(map(lambda x: x + [append_ele], idx))
    else:
        return list(map(lambda x: [append_ele] + x, idx))


def batch_lens(idx: List[List]) -> List:
    return list(map(len, idx))


def as_batch(idx: List) -> List[List]:
    return [idx]


def flatten_lst(lst: List[List]) -> List:
    return [i for sub_lst in lst for i in sub_lst]
