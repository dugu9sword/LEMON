import torch
import numpy as np
import torch.nn.functional as F
# from buff import focal_loss
# from seq_encoder import FofeSeqEncoder, RNNSeqEncoder
import re
import pdb
from buff import time_record
from attention import *
from buff import *
from dataset import *

log_config("main.txt", "cf")
used_data_set = usable_data_sets[config.use_data_set]
vocab_folder = "dataset/ontonotes4/{}.vocab.{}.{}.{}".format(
    config.use_data_set,
    config.char_count_gt, config.bichar_count_gt, config.pos_bmes)
gen_vocab(data_path=used_data_set[0],
          out_folder=vocab_folder,
          char_count_gt=config.char_count_gt,
          bichar_count_gt=config.bichar_count_gt,
          use_cache=config.load_from_cache == "on",
          ignore_tag_bmes=config.pos_bmes == 'off')
gen_lexicon_vocab(*used_data_set,
                  word2vec_path="word2vec/lattice_lstm/ctb.50d.vec",
                  out_folder=vocab_folder,
                  use_cache=config.load_from_cache == "on")

char2idx, idx2char = load_vocab("{}/char.vocab".format(vocab_folder))
bichar2idx, idx2bichar = load_vocab("{}/bichar.vocab".format(vocab_folder))
seg2idx, idx2seg = load_vocab("{}/seg.vocab".format(vocab_folder))
pos2idx, idx2pos = load_vocab("{}/pos.vocab".format(vocab_folder))
ner2idx, idx2ner = load_vocab("{}/ner.vocab".format(vocab_folder))
label2idx, idx2label = load_vocab("{}/label.vocab".format(vocab_folder))
lexicon2idx, idx2lexicon = load_vocab("{}/lexicon.vocab".format(vocab_folder))

with time_record():
    ConllDataSet(
        data_path=used_data_set[1],
        lexicon2idx=lexicon2idx,
        char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx,
        pos2idx=pos2idx, ner2idx=ner2idx, label2idx=label2idx,
        # max_text_len=config.max_sentence_length,
        # max_span_len=config.max_span_length,
        ignore_pos_bmes=config.pos_bmes == 'off',
        sort_by_length=False)
    # dev_set = auto_create(
    #     "dev_set.{}.{}".format(config.use_data_set, config.match_mode),
    #     lambda: ConllDataSet(
    #         data_path=used_data_set[1],
    #         lexicon2idx=lexicon2idx,
    #         char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx,
    #         pos2idx=pos2idx, ner2idx=ner2idx, label2idx=label2idx,
    #         # max_text_len=config.max_sentence_length,
    #         # max_span_len=config.max_span_length,
    #         ignore_pos_bmes=config.pos_bmes == 'off',
    #         sort_by_length=False), cache=config.load_from_cache == "on") # type: ConllDataSet

    # lexmatches = dev_set.data[0].lexmatches
    # for ele in lexmatches:
    #     print("".join(dev_set.data[0].chars[ele[0][0]: ele[0][1] + 1]))
    #     for word_idx, match_type in ele[1]:
    #         print(dev_set.idx2word[word_idx])
pass
