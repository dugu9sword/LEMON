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
