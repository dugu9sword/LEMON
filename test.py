import torch
import numpy as np
import torch.nn.functional as F
# from buff import focal_loss
# from seq_encoder import FofeSeqEncoder, RNNSeqEncoder
import re
import pdb
from buff import time_record
from attention import *

# from transformer.sub_layers import ScaledDotProductAttention, MultiHeadAttention

# from dataset_v2 import load_sentences
#
# import torch
# pdb.set_trace()

# from torch_crf import CRF
#
# seq_length, batch_size, num_tags = 3, 2, 5
# emissions = torch.randn(seq_length, batch_size, num_tags)
# tags = torch.tensor([
#     [0, 1], [2, 4], [3, 1]
# ], dtype=torch.long)  # (seq_length, batch_size)
# mask = torch.tensor([
#     [1, 1], [1, 1], [1, 0]
# ], dtype=torch.uint8)  # (seq_length, batch_size)
# model = CRF(num_tags, batch_first=True)
# # print(model(emissions, tags, mask))
#
# print(model(emissions.permute(1, 0, 2), tags.t(), mask.t()))
#
# print(model.decode(emissions, mask))
from typing import NamedTuple, List, Union

Datum = NamedTuple("Datum", [("shit", str), ("fuck", int)])
data = [Datum(fuck=1, shit="2"), Datum(fuck=3, shit="4")]
# print(data[0]["fuck"])

print(Datum._fields)


def group_fields(lst: List[object],
                 keys: Union[str, List[str]] = None,
                 indices: Union[int, List[int]] = None):
    assert keys is None or indices is None
    is_single = False
    if keys:
        if not isinstance(keys, list):
            keys = [keys]
            is_single = True
        indices = []
        for key in keys:
            obj_type = type(lst[0])
            idx = obj_type._fields.index(key)
            indices.append(idx)
    else:
        if not isinstance(indices, list):
            indices = [indices]
            is_single = True
    rets = []
    for idx in indices:
        rets.append(list(map(lambda item: item[idx], lst)))
    if is_single:
        return rets[0]
    else:
        return rets


with time_record():
    print(group_fields(data, indices=[1, 0]))
