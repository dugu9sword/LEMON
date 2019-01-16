import torch
import torch.nn.functional as F
from buff import focal_loss
from seq_encoder import FofeSeqEncoder, RNNSeqEncoder

x = torch.randn(1, 20, 100, requires_grad=True)
# seq_enc = FofeSeqEncoder(0.5)
seq_enc = RNNSeqEncoder('lstm', 100, 100)
out = seq_enc(x)
pred = torch.nn.Linear(100, 1)(out[:, -1, :])
loss = torch.nn.functional.mse_loss(pred, torch.tensor([[1.]]))
print(loss)
loss.backward()
print(x.grad)
