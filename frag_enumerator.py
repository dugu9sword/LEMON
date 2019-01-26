from buff import *
from enum_index import gen_fragment_ids
from seq_encoder import BaseSeqEncoder


class FragmentEnumerator(torch.nn.Module):
    def __init__(self,
                 max_span_len,
                 encoder_cls,
                 encoder_args,
                 fusion='cat'):
        super(FragmentEnumerator, self).__init__()
        self.b2e_encoder = encoder_cls(*encoder_args)  # type: BaseSeqEncoder
        self.e2b_encoder = encoder_cls(*encoder_args)  # type: BaseSeqEncoder
        self.max_span_len = max_span_len
        self.fusion = fusion

    def enumerate_inputs(self, inputs, lengths):
        """
        通过 for 循环枚举每个句子里的所有最长 span，并拼接起来。
        返回 [用 RNN 来扫描的span 数量，max_span_len, dim]
        """
        batch_size, time_steps, size = inputs.size()
        zero_pad = torch.zeros(self.max_span_len, size, device=inputs.device)
        batch_b2e_sub_inputs = []
        batch_e2b_sub_inputs = []
        for it_id, it_input in enumerate(inputs):
            it_len = lengths[it_id]
            b2e_input = torch.cat(
                [it_input.index_select(0, torch.tensor(list(range(it_len)), device=inputs.device)), zero_pad])
            b2e_sub_inputs = torch.stack([b2e_input[i: i + self.max_span_len] for i in range(it_len)])
            e2b_input = torch.cat(
                [it_input.index_select(0, torch.tensor(list(reversed(range(it_len))), device=inputs.device)), zero_pad])
            e2b_sub_inputs = torch.stack([e2b_input[i: i + self.max_span_len] for i in range(it_len)])

            batch_b2e_sub_inputs.append(b2e_sub_inputs)
            batch_e2b_sub_inputs.append(e2b_sub_inputs)

        batch_b2e_sub_inputs = torch.cat(batch_b2e_sub_inputs)
        batch_e2b_sub_inputs = torch.cat(batch_e2b_sub_inputs)

        return batch_b2e_sub_inputs, batch_e2b_sub_inputs

    def forward(self, inputs: torch.Tensor, lengths: list):
        b2e_sub_inputs, e2b_sub_inputs = self.enumerate_inputs(inputs, lengths)

        # flatten 成 [RNN 数量 * max_span_len， dim]
        b2e_sub_outputs = self.b2e_encoder(b2e_sub_inputs)
        b2e_sub_outputs = b2e_sub_outputs.contiguous().view(-1, b2e_sub_outputs.size(2))

        e2b_sub_outputs = self.e2b_encoder(e2b_sub_inputs)
        e2b_sub_outputs = e2b_sub_outputs.contiguous().view(-1, e2b_sub_outputs.size(2))

        b2e_ids, e2b_ids = gen_fragment_ids(lengths, self.max_span_len)

        b2e_fragments = b2e_sub_outputs.index_select(0, torch.tensor(b2e_ids, device=inputs.device))
        e2b_fragments = e2b_sub_outputs.index_select(0, torch.tensor(e2b_ids, device=inputs.device))

        # check size
        row_num = 0
        for length in lengths:
            if length < self.max_span_len:
                row_num += (1 + length) * length / 2
            else:
                row_num += (length + length - self.max_span_len + 1) \
                           * self.max_span_len / 2
        assert row_num == b2e_fragments.size(0)

        if self.fusion == 'cat':
            fragments = torch.cat([b2e_fragments, e2b_fragments], 1)
        elif self.fusion == 'add':
            fragments = (b2e_fragments + e2b_fragments) / np.sqrt(2)
        else:
            return Exception
        return fragments
