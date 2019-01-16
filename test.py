import torch
import torch.nn.functional as F
from buff import focal_loss

inputs = F.log_softmax(torch.randn(100, 4))
print(inputs)
targets = torch.randint(0, 4, (100, ))
print(targets)

print(F.nll_loss(inputs, targets))
print(focal_loss(inputs, targets, gamma=0))