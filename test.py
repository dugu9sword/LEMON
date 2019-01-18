import torch
import numpy as np
import torch.nn.functional as F
# from buff import focal_loss
# from seq_encoder import FofeSeqEncoder, RNNSeqEncoder
import re
import pdb
from buff import time_record

embeds = torch.nn.Embedding(30000, 100)

indices = torch.randint(0, 30000, (5, 100))

fc = torch.nn.Sequential(
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 1)
)

with time_record():
    tensor2d = []
    for i in range(indices.size(0)):
        tensor1d = []
        for j in range(indices.size(1)):
            tensor1d.append(embeds(indices[i][j]))
        tensor1d = torch.stack(tensor1d, dim=0)
        tensor2d.append(tensor1d)
    tensor2d = torch.stack(tensor2d, dim=0)
    fc(tensor2d).mean().backward()
    # print(tensor2d)
    print(tensor2d.size())

with time_record():
    f = embeds(indices)
    fc(f).mean().backward()

    # print(f)
    print(f.size())
