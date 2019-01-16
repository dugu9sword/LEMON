import torch
from seq_encoder import BaseSeqEncoder
from buff import flip_by_length
from enum_index import gen_context_ids


class ContextEnumerator(torch.nn.Module):
    def __init__(self,
                 max_span_len,
                 b2e_encoder,
                 e2b_encoder):
        super(ContextEnumerator, self).__init__()
        self.b2e_encoder = b2e_encoder  # type: BaseSeqEncoder
        self.e2b_encoder = e2b_encoder  # type: BaseSeqEncoder
        self.max_span_len = max_span_len

    def forward(self, inputs, lengths):
        b2e_outputs = self.b2e_encoder(inputs)
        b2e_outputs = b2e_outputs.contiguous().view(-1, b2e_outputs.size(2))

        e2b_outputs = self.e2b_encoder(flip_by_length(inputs, lengths))
        e2b_outputs = e2b_outputs.contiguous().view(-1, e2b_outputs.size(2))

        b2e_ids, e2b_ids = gen_context_ids(lengths, self.max_span_len)

        left_contexts = b2e_outputs.index_select(0, torch.tensor(b2e_ids).to(inputs.device))
        right_contexts = e2b_outputs.index_select(0, torch.tensor(e2b_ids).to(inputs.device))

        return left_contexts, right_contexts
