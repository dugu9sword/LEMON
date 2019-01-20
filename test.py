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

q = torch.randn(3, 5, 100)
k = torch.randn(3, 2, 50)
v = torch.randn(3, 2, 20)

# att = ScaledDotProductAttention(1, 0.0)
m_att = MultiHeadAttention(n_head=1, d_q=100, d_k=50, d_v=20, d_att_k=50,
                           d_att_v=50, d_out=100)

pad_as_one_mask = gen_att_mask((2, 2, 1), 5)

res, prob = m_att(q, k, v, mask=pad_as_one_mask)

print(res.size())

# print(res)
# print(prob)

# x = torch.randint(0, 10, (2, 3, 4)).float()
# print(x)
# mask = torch.tensor([[1, 1, 0], [1, 0, 0]]).unsqueeze(2).repeat(1, 1, 4).byte()
# print(mask)
# print(mask.size())
# y = x.masked_fill(mask, -np.inf)
# # y = x.masked_select(mask)
# print(y)
