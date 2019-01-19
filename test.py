import torch
import numpy as np
import torch.nn.functional as F
# from buff import focal_loss
# from seq_encoder import FofeSeqEncoder, RNNSeqEncoder
import re
import pdb
from buff import time_record

x = torch.randn(5, 4)
s_x = torch.zeros_like(x)
lengths = torch.tensor([1, 3, 4, 2])

s_length, arg_s = lengths.sort()

print(torch.index_select(x, arg_s, dim=0))
# unsorted.scatter_(0, ind, y)
#

print(s_length)
print(arg_s)