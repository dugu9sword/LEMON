import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from buff import load_word2vec, log


class MixEmbedding(torch.nn.Module):
    def __init__(self,
                 char_vocab_size, char_emb_size,
                 bichar_vocab_size, bichar_emb_size,
                 seg_vocab_size, seg_emb_size):
        super(MixEmbedding, self).__init__()
        self.char_embeds = torch.nn.Embedding(char_vocab_size, char_emb_size)
        if bichar_emb_size > 0:
            self.bichar_embeds = torch.nn.Embedding(bichar_vocab_size, bichar_emb_size)
        else:
            self.bichar_embeds = None
        self.seg_embeds = torch.nn.Embedding(seg_vocab_size, seg_emb_size)

    @property
    def embedding_dim(self):
        if self.bichar_embeds:
            return self.char_embeds.embedding_dim + self.bichar_embeds.embedding_dim + self.seg_embeds.embedding_dim
        else:
            return self.char_embeds.embedding_dim + self.seg_embeds.embedding_dim

    def forward(self, pad_chars, pad_bichars, pad_segs):
        if self.bichar_embeds:
            final_embs = torch.cat([self.char_embeds(pad_chars),
                                    self.seg_embeds(pad_segs),
                                    self.bichar_embeds(pad_bichars)], dim=2)
        else:
            final_embs = torch.cat([self.char_embeds(pad_chars),
                                    self.seg_embeds(pad_segs)], dim=2)
        return final_embs

    def show_mean_std(self):
        log("Embedding Info")
        log("\t[char] mean {} std {}".format(
            torch.mean(self.char_embeds.weight),
            torch.std(self.char_embeds.weight),
        ))
        if self.bichar_embeds:
            log("\t[bichar] mean {} std {}".format(
                torch.mean(self.bichar_embeds.weight),
                torch.std(self.bichar_embeds.weight),
            ))
        log("\t[seg] mean {} std {}".format(
            torch.mean(self.seg_embeds.weight),
            torch.std(self.seg_embeds.weight),
        ))


class BiRNNTokenEncoder(torch.nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, num_layers, dropout):
        super(BiRNNTokenEncoder, self).__init__()
        RNN: torch.nn.Module
        if cell_type == 'lstm':
            RNN = torch.nn.LSTM
        elif cell_type == 'gru':
            RNN = torch.nn.GRU
        self.encoder = RNN(input_size=input_size,
                           hidden_size=hidden_size // 2,
                           num_layers=num_layers,
                           bidirectional=True,
                           batch_first=True,
                           dropout=dropout)

    def forward(self, input_embs, text_lens):
        packed_input_embs = pack_padded_sequence(input_embs, text_lens, batch_first=True)
        rnn_out, _ = self.encoder(packed_input_embs)
        token_reps, _ = pad_packed_sequence(rnn_out, batch_first=True)
        return token_reps
