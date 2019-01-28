import torch
import numpy as np
import torch.nn.functional as F
# from buff import focal_loss
# from seq_encoder import FofeSeqEncoder, RNNSeqEncoder
import re
import pdb
from buff import time_record
from attention import *
from buff import *

from typing import NamedTuple

Pair = NamedTuple("Pair", [("a", object), ("b", object)])
def fragments(sentence_len, max_span_len) -> List:
    ret = []
    for i in range(sentence_len):
        for j in range(i, i + max_span_len):
            if j == sentence_len:
                break
            ret.append((i, j))
    return ret


print(fragments(4, 10))