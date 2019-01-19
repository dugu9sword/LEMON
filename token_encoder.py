import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from buff import load_word2vec, log


class MixEmbedding(torch.nn.Module):
    def __init__(self,
                 char_vocab_size, char_emb_size,
                 bichar_vocab_size, bichar_emb_size,
                 seg_vocab_size, seg_emb_size,
                 pos_vocab_size, pos_emb_size):
        super(MixEmbedding, self).__init__()
        self.char_embeds = torch.nn.Embedding(char_vocab_size, char_emb_size)
        if bichar_emb_size > 0:
            self.bichar_embeds = torch.nn.Embedding(bichar_vocab_size, bichar_emb_size)
        else:
            self.bichar_embeds = None
        if seg_emb_size > 0:
            self.seg_embeds = torch.nn.Embedding(seg_vocab_size, seg_emb_size)
        else:
            self.seg_embeds = None
        if pos_emb_size > 0:
            self.pos_embeds = torch.nn.Embedding(pos_vocab_size, pos_emb_size)
        else:
            self.pos_embeds = None

    @property
    def embedding_dim(self):
        ret = self.char_embeds.embedding_dim
        if self.bichar_embeds:
            ret += self.bichar_embeds.embedding_dim
        if self.seg_embeds:
            ret += self.seg_embeds.embedding_dim
        if self.pos_embeds:
            ret += self.pos_embeds.embedding_dim
        return ret

    def forward(self, pad_chars, pad_bichars, pad_segs, pad_poss):
        embeds_to_cat = [self.char_embeds(pad_chars)]
        if self.seg_embeds:
            embeds_to_cat.append(self.seg_embeds(pad_segs))
        if self.pos_embeds:
            embeds_to_cat.append(self.pos_embeds(pad_poss))
        if self.bichar_embeds:
            embeds_to_cat.append(self.bichar_embeds(pad_bichars))
        final_embs = torch.cat(embeds_to_cat, dim=2)
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
        if self.seg_embeds:
            log("\t[seg] mean {} std {}".format(
                torch.mean(self.seg_embeds.weight),
                torch.std(self.seg_embeds.weight),
            ))
        if self.pos_embeds:
            log("\t[pos] mean {} std {}".format(
                torch.mean(self.pos_embeds.weight),
                torch.std(self.pos_embeds.weight),
            ))


class BiRNNTokenEncoder(torch.nn.Module):
    def __init__(self, cell_type, input_size, hidden_size, num_layers,
                 dropout):
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
