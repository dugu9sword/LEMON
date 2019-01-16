import torch


class BaseSeqEncoder(torch.nn.Module):
    def forward(self, inputs):
        raise NotImplemented


class PlainSeqEncoder(BaseSeqEncoder):
    def forward(self, inputs):
        return inputs


class FofeSeqEncoder(BaseSeqEncoder):
    def __init__(self, alpha):
        super(FofeSeqEncoder, self).__init__()
        self.alpha = alpha

    def forward(self, inputs, batch_first=True):
        if batch_first:
            inputs = inputs.transpose(0, 1)
        time_steps, batch_size, size = inputs.size()
        output = torch.zeros(batch_size, size).to(inputs.device)
        outputs = []
        for t, t_input in enumerate(inputs):
            output = output * self.alpha + t_input
            outputs.append(output)
        outputs = torch.stack(outputs)
        if batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs


class AverageSeqEncoder(BaseSeqEncoder):
    def __init__(self):
        super(AverageSeqEncoder, self).__init__()

    def forward(self, inputs, batch_first=True):
        if batch_first:
            inputs = inputs.transpose(0, 1)
        time_steps, batch_size, size = inputs.size()
        output = torch.zeros(batch_size, size).to(inputs.device)
        outputs = []
        for t, t_input in enumerate(inputs):
            output = (output * t + t_input) / (t + 1)
            outputs.append(output)
        outputs = torch.stack(outputs)
        if batch_first:
            outputs = outputs.transpose(0, 1)
        return outputs


class RNNSeqEncoder(BaseSeqEncoder):
    def __init__(self, cell, input_size, hidden_size):
        super(RNNSeqEncoder, self).__init__()
        if cell == 'gru':
            self.rnn = torch.nn.GRU(input_size=input_size,
                                    hidden_size=hidden_size,
                                    batch_first=True)
        elif cell == 'lstm':
            self.rnn = torch.nn.LSTM(input_size=input_size,
                                     hidden_size=hidden_size,
                                     batch_first=True)
        else:
            raise Exception('cell type not specified')

    def forward(self, inputs):
        outputs, _ = self.rnn(inputs)
        return outputs
