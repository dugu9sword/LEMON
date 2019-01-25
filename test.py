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

log_config("test", "cf")
log_to_buffer("ff")
log_to_buffer("ff")
log_to_buffer("ff")
log_flush_buffer()

embeds = torch.nn.Embedding(5, 2, sparse=True)

print(embeds.weight)