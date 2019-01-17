import torch
from seq_encoder import BaseSeqEncoder
from buff import flip_by_length
from enum_index import gen_inc_context_ids, gen_exc_context_ids


class ContextEnumerator(torch.nn.Module):
    def __init__(self,
                 max_span_len,
                 b2e_encoder,
                 e2b_encoder,
                 out_size,
                 include=True):
        super(ContextEnumerator, self).__init__()
        self.b2e_encoder = b2e_encoder  # type: BaseSeqEncoder
        self.e2b_encoder = e2b_encoder  # type: BaseSeqEncoder
        self.max_span_len = max_span_len
        self.out_size = out_size
        self.include = include
        if not include:
            self.b_start_tensor = torch.nn.Parameter(torch.Tensor(1, out_size))
            self.e_start_tensor = torch.nn.Parameter(torch.Tensor(1, out_size))
            torch.nn.init.xavier_normal_(self.b_start_tensor)
            torch.nn.init.xavier_normal_(self.e_start_tensor)

    def forward(self, inputs, lengths):
        b2e_outputs = self.b2e_encoder(inputs)
        b2e_outputs = b2e_outputs.contiguous().view(-1, b2e_outputs.size(2))

        e2b_outputs = self.e2b_encoder(flip_by_length(inputs, lengths))
        e2b_outputs = e2b_outputs.contiguous().view(-1, e2b_outputs.size(2))

        if self.include:
            b2e_ids, e2b_ids = gen_inc_context_ids(lengths, self.max_span_len)
            left_contexts = b2e_outputs.index_select(0, torch.tensor(b2e_ids).to(inputs.device))
            right_contexts = e2b_outputs.index_select(0, torch.tensor(e2b_ids).to(inputs.device))
        else:
            b2e_ids, e2b_ids = gen_exc_context_ids(lengths, self.max_span_len)

            pad_index = b2e_outputs.size(0)
            b2e_ids = [ele if ele != -1 else pad_index for ele in b2e_ids]
            e2b_ids = [ele if ele != -1 else pad_index for ele in e2b_ids]

            b2e_outputs = torch.cat([b2e_outputs, self.b_start_tensor], dim=0)
            e2b_outputs = torch.cat([e2b_outputs, self.e_start_tensor], dim=0)

            # print(self.b_start_tensor[0][:10])
            # print(b2e_outputs[0][:10])

            left_contexts = b2e_outputs.index_select(0, torch.tensor(b2e_ids).to(inputs.device))
            right_contexts = e2b_outputs.index_select(0, torch.tensor(e2b_ids).to(inputs.device))

        return left_contexts, right_contexts
