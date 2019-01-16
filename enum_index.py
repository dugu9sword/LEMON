from functools import lru_cache
from typing import List

"""
对于序列 [ABCD, EF], 他们的片段按照如下排序：
    A, AB, ABC, B, BC, BCD, C, CD, D, E, EF, F
"""


def gen_inc_context_ids(lengths=(4, 2), max_span_len=3):
    """
    对于 [ABCD, EF]，用 RNN 从两个方向扫描，RNN 个数为 2，正反向分别为
            -->: Tensor(2, 4, dim)
                    ABCD, EF**
                       ^: left inclusive context for [BCD]
                          1d index    : 4 <-- 该句子 RNN 的 4 个 hidden
                          flatten idx : 4
            <--: Tensor(2, 4, dim)
                    DCBA, FE**
                           ^:  right inclusive context for [FE]
                               1d index    : 2
                               flatten idx : 6
    将其 flatten 成 (8, dim)，用于后续的 index_select
    """
    max_len = max(lengths)
    b2e_ids, e2b_ids = [], []
    base_row = 0
    for length in lengths:
        it_b2e_1d, it_e2b_1d = _gen_inc_context_ids(length, max_span_len)
        b2e_ids.extend(list(map(lambda x: x + base_row, it_b2e_1d)))
        e2b_ids.extend(list(map(lambda x: x + base_row, it_e2b_1d)))
        base_row += max_len
    return b2e_ids, e2b_ids


@lru_cache(maxsize=None)
def _gen_inc_context_ids(length=4, max_span_len=3):
    # 保存 i 代表第 i 个 hidden state
    b2e_1d, e2b_1d = [], []
    for i in range(length):
        for j in range(max_span_len):
            if i + j < length:
                b2e_1d.append(i + j)
                e2b_1d.append(length - i - 1)
    return b2e_1d, e2b_1d


def gen_exc_context_ids(lengths=(4, 2), max_span_len=3):
    """
    对于 [ABCD, EF]，用 RNN 从两个方向扫描，RNN 个数为 2，正反向分别为
            -->: Tensor(2, 4, dim)
                    ABCD, EF**
                     ^: left exclusive context for [CD]
                         1d index    : 1 <-- 该句子 RNN 的 1 个 hidden
                         flatten idx : 1
            <--: Tensor(2, 4, dim)
                    DCBA, FE**
                        ^:  right exclusive context for [CBA]
                         1d index    : -1
                         flatten idx : -1
    将其 flatten 成 (8, dim)，用于后续的 index_select
    """
    max_len = max(lengths)
    b2e_ids, e2b_ids = [], []
    base_row = 0
    for length in lengths:
        it_b2e_1d, it_e2b_1d = _gen_exc_context_ids(length, max_span_len)
        b2e_ids.extend(list(map(lambda x: x + base_row if x != -1 else -1, it_b2e_1d)))
        e2b_ids.extend(list(map(lambda x: x + base_row if x != -1 else -1, it_e2b_1d)))
        base_row += max_len
    return b2e_ids, e2b_ids


def _gen_exc_context_ids(length=4, max_span_len=3):
    # 保存 i 代表第 i 个 hidden state
    b2e_1d, e2b_1d = [], []
    for i in range(length):
        for j in range(max_span_len):
            if i + j < length:
                b2e_1d.append(i - 1)
                e2b_1d.append(length - 1 - i - j - 1)
    return b2e_1d, e2b_1d


# x, y = gen_context_ids((4, 2, 3), 3)
# print(x)
# print(y)

# print(_gen_exc_context_ids())
# print(gen_exc_context_ids())


def gen_fragment_ids(lengths=(4, 2), max_span_len=3):
    """
    对于 [ABCD, EF]，用窗口为 3 的 RNN 扫描，RNN 个数为 n，正反向分别为
        -->: Tensor(6, 3, dim)
                ABC, BCD, CD*, D**, EF*, F**
                 |                   ^: hidden state for [EF]
                 |                      2d index    : (0, 1)
                 |                      flatten idx : 13
                 ^: hidden state for [AB]
                    2d index    : (0, 1) <-- 该句子的第 0 个 RNN 而非 6 个 RNN 里的第 0 个
                    flatten idx : 1
        <--: Tensor(6, 3, dim)
                DCB, CBA, BA*, A**, FE*, E**
                           ^: hidden state for [BA]
                              2d index    : (0, 2)
                              flatten idx : 7
    将其 flatten 成 (18, dim)，用于后续的 index_select
    """
    b2e_ids, e2b_ids = [], []
    base_row = 0
    for length in lengths:
        b2e_2d, e2b_2d = _gen_fragment_ids(length, max_span_len)
        it_b2e_idx = list(map(lambda pair: pair[0] * max_span_len + pair[1], b2e_2d))
        it_e2b_idx = list(map(lambda pair: pair[0] * max_span_len + pair[1], e2b_2d))
        b2e_ids.extend(list(map(lambda x: x + base_row, it_b2e_idx)))
        e2b_ids.extend(list(map(lambda x: x + base_row, it_e2b_idx)))
        base_row += length * max_span_len
    return b2e_ids, e2b_ids


@lru_cache(maxsize=None)
def _gen_fragment_ids(length=4, max_span_len=3):
    # 保存 (i, j) 代表该句子的第 i 个 RNN 的第 j 个 hidden state
    b2e_2d, e2b_2d = [], []
    for i in range(length):
        for j in range(max_span_len):
            if i + j < length:
                b2e_2d.append((i, j))
                e2b_2d.append((length - i - j - 1, j))
    return b2e_2d, e2b_2d
