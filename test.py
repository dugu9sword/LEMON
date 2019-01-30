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
import collections
from typing import NamedTuple

Pair = NamedTuple("Pair", [("a", int), ("b", int)])
Fuck = NamedTuple("Fuck", [("pair", Pair), ("b", List[int])])

Pair2 = collections.namedtuple("Pair2", ["a", "b"])
Fuck2 = collections.namedtuple("Fuck2", ["pair", "b"])

NUM = 100000

x = []
for _ in range(NUM):
    x.append(Fuck(Pair(1, 2), [1, 2, 3, 4]))
save_var(x, "fuckx")

y = []
for _ in range(NUM):
    y.append(((1, 2), [1, 2, 3, 4]))
save_var(y, "fucky")

z = []
for _ in range(NUM):
    z.append(Fuck2(Pair2(1, 2), [1, 2, 3, 4]))
save_var(z, "fuckz")

with time_record():
    load_var("fuckx")

with time_record():
    load_var("fucky")

with time_record():
    load_var("fuckz")
