# import torch
# import torch.nn.functional as F
# from buff import focal_loss
# from seq_encoder import FofeSeqEncoder, RNNSeqEncoder
import re
import pdb

found_tag = re.search(r'\(([^\s]+)\s+{}\)'.format("\?"),
                      "(NN ?)")
print(found_tag)
