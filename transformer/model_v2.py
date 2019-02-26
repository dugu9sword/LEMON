from transformer.helper import *
from transformer.sub_layers import *
from functools import lru_cache


@lru_cache(maxsize=None)
def position_idx(seq_len, max_len):
    return list(range(1, seq_len + 1)) + [0 for _ in range(max_len - seq_len)]


class TransformerEncoderV2(nn.Module):
    def __init__(
            self,
            d_model,
            d_head, n_head,
            len_max_seq, n_layers,
            dropout=0.1):

        super().__init__()

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(len_max_seq + 1,
                                        d_model,
                                        padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model,
                         d_inner=2 * d_model,
                         n_head=n_head,
                         d_k=d_head,
                         d_v=d_head,
                         dropout=dropout)
            for _ in range(n_layers)])

    def gen_masks(self, pad_chars):
        return {
            "slf_attn_mask": get_attn_key_pad_mask(seq_k=pad_chars, seq_q=pad_chars),
            "non_pad_mask": get_non_pad_mask(pad_chars)
        }

    def forward(self, input_embs, masks, text_lens, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = masks["slf_attn_mask"]
        non_pad_mask = masks["non_pad_mask"]

        # -- Forward
        # self.src_word_emb(src_seq)

        position_tensor = torch.tensor(
            list(map(lambda x: position_idx(x, text_lens[0]), text_lens))
        ).to(input_embs.device)

        enc_output = input_embs + self.position_enc(position_tensor)

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionWiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask
        return enc_output, enc_slf_attn
