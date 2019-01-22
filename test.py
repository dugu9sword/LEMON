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

from dataset_v2 import load_sentences


import torch
# pdb.set_trace()