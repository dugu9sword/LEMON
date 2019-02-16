# -*- coding: UTF-8 -*-

from buff import *
from torch.nn.utils import clip_grad_norm_
import torch
from buff import focal_loss, group_fields
import torch.nn.functional as F
from functools import lru_cache
from dataset import ConllDataSet, gen_lexicon_vocab, load_vocab, gen_vocab, usable_data_sets, match2idx_naive
from program_args import config
from model import Luban7
from evaluation import CRFEvaluator, LubanEvaluator, LubanSpan, luban_span_to_str
import pdb

device = allocate_cuda_device(0)


@lru_cache(maxsize=None)
def enum_span_by_length(text_len):
    span_lst = []
    for begin_token_idx in range(text_len):
        for span_len in range(1, config.max_span_length + 1):
            end_token_idx = begin_token_idx + span_len - 1
            if end_token_idx < text_len:
                span_lst.append((begin_token_idx, end_token_idx))  # tuple
    return span_lst


###################################################################
# Main
###################################################################

def main():
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

    idx2str = lambda idx_lst: "".join(map(lambda x: idx2char[x], idx_lst))
    train_set = auto_create(
        "train_set.{}.{}".format(config.use_data_set, config.match_mode),
        lambda: ConllDataSet(
            data_path=used_data_set[0],
            lexicon2idx=lexicon2idx,
            char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx,
            pos2idx=pos2idx, ner2idx=ner2idx, label2idx=label2idx,
            ignore_pos_bmes=config.pos_bmes == 'off',
            max_text_len=config.max_sentence_length,
            max_span_len=config.max_span_length,
            sort_by_length=True), cache=config.load_from_cache == "on")  # type: ConllDataSet
    dev_set = auto_create(
        "dev_set.{}.{}".format(config.use_data_set, config.match_mode),
        lambda: ConllDataSet(
            data_path=used_data_set[1],
            lexicon2idx=lexicon2idx,
            char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx,
            pos2idx=pos2idx, ner2idx=ner2idx, label2idx=label2idx,
            # max_text_len=config.max_sentence_length,
            # max_span_len=config.max_span_length,
            ignore_pos_bmes=config.pos_bmes == 'off',
            sort_by_length=False), cache=config.load_from_cache == "on")  # type: ConllDataSet
    test_set = auto_create(
        "test_set.{}.{}".format(config.use_data_set, config.match_mode),
        lambda: ConllDataSet(
            data_path=used_data_set[2],
            lexicon2idx=lexicon2idx,
            char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx,
            pos2idx=pos2idx, ner2idx=ner2idx, label2idx=label2idx,
            # max_text_len=config.max_sentence_length,
            # max_span_len=config.max_span_length,
            ignore_pos_bmes=config.pos_bmes == 'off',
            sort_by_length=False), cache=config.load_from_cache == "on")  # type: ConllDataSet
    longest_span_len = max(train_set.longest_span_len, dev_set.longest_span_len)
    longest_text_len = max(train_set.longest_text_len, dev_set.longest_text_len)

    luban7 = Luban7(char2idx=char2idx,
                    bichar2idx=bichar2idx,
                    seg2idx=seg2idx,
                    pos2idx=pos2idx,
                    ner2idx=ner2idx,
                    label2idx=label2idx,
                    longest_text_len=longest_text_len,
                    lexicon2idx=lexicon2idx).to(device)

    # set optimizer
    optimizers = []
    lr_scls = []
    if config.opt_type == "adam":
        if config.use_sparse_embed == "on":
            params = list(luban7.named_parameters())
            dense_params, sparse_params = [], []
            for pid in range(len(params)):
                if "embeds" in params[pid][0]:
                    sparse_params.append(params[pid][1])
                else:
                    dense_params.append(params[pid][1])
            optimizers.append(torch.optim.Adam(dense_params, lr=config.lr, weight_decay=config.weight_decay))
            optimizers.append(torch.optim.SparseAdam(sparse_params, lr=config.lr))
        else:
            optimizers.append(torch.optim.Adam(luban7.parameters(), lr=config.lr, weight_decay=config.weight_decay))
    elif config.opt_type == "sgd":
        optimizers.append(torch.optim.SGD(luban7.parameters(), lr=config.lr, weight_decay=config.weight_decay,
                                          momentum=config.momentum))
    elif config.opt_type == "adadelta":
        optimizers.append(torch.optim.Adadelta(luban7.parameters(), lr=config.lr,
                                               weight_decay=config.weight_decay))
    else:
        raise Exception
    for opt in optimizers:
        lr_scls.append(torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=list(range(config.lr_epoch, config.epoch_max)),
            gamma=config.lr_gamma))

    manager = ModelManager(luban7, config.model_name, init_ckpt=config.model_ckpt) \
        if config.model_name != "off" else None

    epoch_id = -1
    while True:
        """
        Epoch Level Pre-processing
        """
        for lr_scl in lr_scls:
            lr_scl.step()
        epoch_id += 1
        if epoch_id == config.epoch_max:
            break
        # luban7.embeds.fix_grad(epoch_id < config.epoch_fix_char_emb)
        if config.match_mode != 'off':
            for param in luban7.lexicon_embeds.parameters():
                param.requires_grad = epoch_id > config.epoch_fix_lexicon_emb
        for param in luban7.embeds.parameters():
            param.requires_grad = epoch_id > config.epoch_fix_char_emb
        luban7.embeds.show_mean_std()

        """
        Training
        """
        crf_evaluator = CRFEvaluator(idx2tag=idx2ner)
        if config.train_on == "on":
            log(">>> epoch {} train".format(epoch_id))
            luban7.train()
            train_set.reset(shuffle=True)
            iter_id = 0
            progress = ProgressManager(total=train_set.size)
            log(train_set.size)
            while not train_set.finished:
                iter_id += 1
                batch_data = train_set.next_batch(config.batch_size)
                batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)

                # >>> CRF
                if config.crf == 0.0:
                    crf_loss = 0.0
                    crf_log = "no crf"
                else:
                    crf_loss = luban7.crf_nll(batch_data)
                    crf_f1 = crf_evaluator.eval(luban7.crf_decode(batch_data),
                                                group_fields(batch_data, "ners"))[2]
                    crf_log = "crf loss: {:.4f} f1: {:.4f} ".format(crf_loss.item(), crf_f1)
                # <<< CRF

                # >>> Luban
                if config.crf == 1.0:
                    luban_loss = 0
                    luban_log = "no luban"
                else:
                    score, span_ys = luban7.get_span_score_tags(batch_data)
                    luban_loss = focal_loss(inputs=score,
                                            targets=torch.tensor(span_ys, device=device),
                                            gamma=config.focal_gamma)
                    score_probs = F.softmax(score, dim=1)
                    luban_precision = accuracy(score_probs.detach().cpu().numpy(), span_ys)
                    luban_log = "luban loss: {:.4f} precision: {:.4f}".format(luban_loss.item(), luban_precision)
                # <<< Luban

                loss = config.crf * crf_loss + (1 - config.crf) * luban_loss

                progress.update(len(batch_data))
                log(
                    "[{}: {}/{}] ".format(epoch_id, progress.complete_num, train_set.size),
                    "b: {:.2f} / c:{:.2f} / r: {:.2f} ".format(
                        progress.batch_time, progress.cost_time, progress.rest_time),
                    crf_log, luban_log
                )

                # update gradients
                for opt in optimizers:
                    opt.zero_grad()
                loss.backward()
                if config.check_nan == "on":
                    if torch.isnan(luban7.embeds.char_embeds.weight.grad.sum()):
                        pdb.set_trace()
                clip_grad_norm_(luban7.parameters(), 5)
                for opt in optimizers:
                    opt.step()

            log("<<< epoch {} train".format(epoch_id))

        if isinstance(manager, ModelManager):
            manager.save()

        """
        Development
        """
        thresholds = [-1, 0.1, 0.2, 0.3, 0.4]
        crf_evaluator = CRFEvaluator(idx2tag=idx2ner)
        luban_evals = [LubanEvaluator() for _ in range(len(thresholds))]
        with torch.no_grad():
            luban7.eval()
            sets_for_validation = {"dev_set": dev_set, "test_set": test_set}
            if epoch_id > config.epoch_show_train:
                sets_for_validation["train_set"] = train_set
            for set_name, set_for_validation in sets_for_validation.items():
                log(">>> epoch {} validation on {}".format(epoch_id, set_name))

                set_for_validation.reset(shuffle=False)
                progress = ProgressManager(total=set_for_validation.size)
                while not set_for_validation.finished:
                    batch_data = set_for_validation.next_batch(10, fill_batch=True)
                    batch_data = sorted(batch_data, key=lambda x: len(x[0]), reverse=True)  # type: List[Datum]

                    texts = list(map(lambda x: x[0], batch_data))
                    text_lens = batch_lens(texts)

                    # >>> CRF
                    if config.crf != 0.0:
                        results = luban7.crf_decode(batch_data)
                        crf_evaluator.eval(results, group_fields(batch_data, "ners"))

                    # <<< CRF

                    # >>> Luban
                    if config.crf != 1.0:
                        score, span_ys = luban7.get_span_score_tags(batch_data)
                        score_probs = F.softmax(score, dim=1)

                        for ts_id, threshold in enumerate(thresholds):
                            if threshold == -1:
                                pred = cast_list(torch.argmax(score_probs, 1))
                            else:
                                ner_score_probs = score_probs.clone()
                                ner_score_probs[:, 0] = threshold
                                pred = cast_list(torch.argmax(ner_score_probs, 1))

                            offset = 0
                            for bid in range(len(text_lens)):
                                log_to_buffer("[{:>4}] [ ts = {:>.1f} ] {}".format(
                                    progress.complete_num + bid, threshold,
                                    idx2str(batch_data[bid].chars)))
                                enum_spans = enum_span_by_length(text_lens[bid])
                                # fragment_score = score[offset: offset + len(enum_spans)]
                                # log(fragment_score)
                                luban_spans = []
                                for sid, span in enumerate(enum_spans):
                                    begin_idx, end_idx = span
                                    span_offset = sid + offset
                                    if pred[span_offset] != 0 or span_ys[span_offset] != 0:
                                        luban_span = LubanSpan(
                                            bid=begin_idx, eid=end_idx, lid=pred[span_offset],
                                            pred_prob=score_probs[span_offset][pred[span_offset]],
                                            gold_prob=score_probs[span_offset][span_ys[span_offset]],
                                            pred_label=idx2label[pred[span_offset]],
                                            gold_label=idx2label[span_ys[span_offset]],
                                            fragment=idx2str(batch_data[bid].chars[begin_idx: end_idx + 1])
                                        )
                                        luban_spans.append(luban_span)
                                        log_to_buffer(luban_span_to_str(luban_span))
                                luban_evals[ts_id].decode(luban_spans)
                                offset += len(enum_spans)
                            log_flush_buffer()

                    # <<< Luban

                    progress.update(len(batch_data))

                log("** result.crf epoch {} on {}: precision {:.4f}, recall {:.4f}, f1 {:.4f}".format(
                    epoch_id, set_name, *crf_evaluator.prf))
                for ts_id in range(len(thresholds)):
                    log("** result.luban epoch {}[threshold{:.2f}] on {}: precision {:.4f}, recall {:.4f}, f1 {:.4f}".format(
                        epoch_id, thresholds[ts_id], set_name, *luban_evals[ts_id].prf))
                log("<<< epoch {} validation on {}".format(epoch_id, set_name))

        """
        Epoch post-processing
        """
        pass


if __name__ == '__main__':
    main()
