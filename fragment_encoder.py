from buff import *
from functools import lru_cache


class BaseFragEncoder(torch.nn.Module):
    def forward(self, inputs):
        raise NotImplemented


class PlainFragEncoder(BaseFragEncoder):
    def forward(self, inputs):
        return inputs


class FofeFragEncoder(BaseFragEncoder):
    def __init__(self, alpha):
        super(FofeFragEncoder, self).__init__()
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


class AverageFragEncoder(BaseFragEncoder):
    def __init__(self):
        super(AverageFragEncoder, self).__init__()

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


class RNNFragEncoder(BaseFragEncoder):
    def __init__(self, cell, input_size, hidden_size):
        super(RNNFragEncoder, self).__init__()
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


class AllFragEncoder(torch.nn.Module):
    def __init__(self,
                 max_span_len,
                 b2e_encoder,
                 e2b_encoder):
        super(AllFragEncoder, self).__init__()
        self.b2e_encoder = b2e_encoder
        self.e2b_encoder = e2b_encoder
        self.max_span_len = max_span_len

    def forward(self, inputs: torch.Tensor, lengths: list):
        batch_size, time_steps, size = inputs.size()
        zero_pad = torch.zeros(self.max_span_len, size).to(inputs.device)
        batch_b2e_sub_inputs = []
        batch_e2b_sub_inputs = []
        for it_id, it_input in enumerate(inputs):
            it_len = lengths[it_id]
            b2e_input = torch.cat(
                [it_input.index_select(0, torch.tensor(list(range(it_len))).to(inputs.device)), zero_pad])
            b2e_sub_inputs = torch.stack([b2e_input[i: i + self.max_span_len] for i in range(it_len)])
            e2b_input = torch.cat(
                [it_input.index_select(0, torch.tensor(list(reversed(range(it_len)))).to(inputs.device)), zero_pad])
            e2b_sub_inputs = torch.stack([e2b_input[i: i + self.max_span_len] for i in range(it_len)])

            batch_b2e_sub_inputs.append(b2e_sub_inputs)
            batch_e2b_sub_inputs.append(e2b_sub_inputs)

        batch_b2e_sub_inputs = torch.cat(batch_b2e_sub_inputs)
        batch_b2e_sub_outputs = self.b2e_encoder(batch_b2e_sub_inputs)
        batch_b2e_sub_outputs = batch_b2e_sub_outputs.contiguous() \
            .view(-1, batch_b2e_sub_outputs.size(2))

        batch_e2b_sub_inputs = torch.cat(batch_e2b_sub_inputs)
        batch_e2b_sub_outputs = self.e2b_encoder(batch_e2b_sub_inputs)
        batch_e2b_sub_outputs = batch_e2b_sub_outputs.contiguous() \
            .view(-1, batch_e2b_sub_outputs.size(2))

        b2e_ids, e2b_ids = AllFragEncoder.fragment_index(lengths, self.max_span_len)

        b2e_fragments = batch_b2e_sub_outputs.index_select(0, torch.tensor(b2e_ids).to(inputs.device))
        e2b_fragments = batch_e2b_sub_outputs.index_select(0, torch.tensor(e2b_ids).to(inputs.device))

        # check size
        row_num = 0
        for length in lengths:
            if length < self.max_span_len:
                row_num += (1 + length) * length / 2
            else:
                row_num += (length + length - self.max_span_len + 1) \
                           * self.max_span_len / 2
        assert row_num == b2e_fragments.size(0)

        fragments = torch.cat([b2e_fragments, e2b_fragments], 1)
        return fragments

    @staticmethod
    def fragment_index(lengths=[4, 2], max_span_len=3):
        """
        given two sequences [ABCD**, EF**], and its {b->e|e<-b}fragments:
            b->e: ABC, BCD, CD*, D**, EF*, F**
            e->b: DCB, CBA, BA*, A**, FE*, E**
        flatten both the fragments and reorder as:
            A, AB, ABC, B, BC, BCD, C, CD, D, E, EF, F
        """
        b2e_ids, e2b_ids = [], []
        base_row = 0
        for length in lengths:
            b2e_2d, e2b_2d = AllFragEncoder.__fragment_index(length, max_span_len)
            it_b2e_idx = list(map(lambda pair: pair[0] * max_span_len + pair[1], b2e_2d))
            it_e2b_idx = list(map(lambda pair: pair[0] * max_span_len + pair[1], e2b_2d))
            b2e_ids.extend(list(map(lambda x: x + base_row, it_b2e_idx)))
            e2b_ids.extend(list(map(lambda x: x + base_row, it_e2b_idx)))
            base_row += length * max_span_len
        return b2e_ids, e2b_ids

    @staticmethod
    @lru_cache(maxsize=None)
    def __fragment_index(length=4, max_span_len=3):
        b2e_2d, e2b_2d = [], []
        for i in range(length):
            for j in range(max_span_len):
                if i + j < length:
                    b2e_2d.append((i, j))
                    e2b_2d.append((length - i - j - 1, j))
        return b2e_2d, e2b_2d


if __name__ == "__main__":
    # TEST AVERAGE
    avg = AverageFragEncoder()
    hi = torch.tensor([[[i, i] for i in range(1, 6)] for j in range(3)]).float()
    log(hi, 'g')
    log(avg(hi), 'b')

    # TEST FRAGMENT ENCODER
    hi = torch.tensor([[[i, i] for i in range(1, 6)] for j in range(3)]).float()
    log(hi, 'g')
    enc = AllFragEncoder(max_span_len=3,
                         b2e_encoder=PlainFragEncoder(),
                         e2b_encoder=PlainFragEncoder())
    log(enc(hi, [5, 4, 2]), 'b')

    enc = AllFragEncoder(max_span_len=3,
                         b2e_encoder=RNNFragEncoder('lstm', 2, 2),
                         e2b_encoder=RNNFragEncoder('lstm', 2, 2))
    log(enc(hi, [5, 4, 2]), 'b')
