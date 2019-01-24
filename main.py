# -*- coding: UTF-8 -*-

from buff import *
from torch.nn.utils import clip_grad_norm_
import torch
from buff import focal_loss, group_fields
import torch.nn.functional as F
from functools import lru_cache
from dataset import ConllDataSet, SpanLabel, SpanPred, load_vocab, gen_vocab, usable_data_sets
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
    vocab_folder = "dataset/ontonotes4/vocab.{}.{}.{}".format(
        config.char_count_gt, config.bichar_count_gt, config.pos_bmes)
    gen_vocab(data_path=used_data_set[0],
              out_folder=vocab_folder,
              char_count_gt=config.char_count_gt,
              bichar_count_gt=config.bichar_count_gt,
              use_cache=config.load_from_cache,
              ignore_tag_bmes=config.pos_bmes == 'off')
    char2idx, idx2char = load_vocab("{}/char.vocab".format(vocab_folder))
    bichar2idx, idx2bichar = load_vocab("{}/bichar.vocab".format(vocab_folder))
    seg2idx, idx2seg = load_vocab("{}/seg.vocab".format(vocab_folder))
    pos2idx, idx2pos = load_vocab("{}/pos.vocab".format(vocab_folder))
    ner2idx, idx2ner = load_vocab("{}/ner.vocab".format(vocab_folder))
    label2idx, idx2label = load_vocab("{}/label.vocab".format(vocab_folder))

    idx2str = lambda idx_lst: "".join(map(lambda x: idx2char[x], idx_lst))
    train_set = auto_create(
        "train_set.{}".format(config.use_data_set),
        lambda: ConllDataSet(
            data_path=used_data_set[0],
            char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx,
            pos2idx=pos2idx, ner2idx=ner2idx, label2idx=label2idx,
            ignore_pos_bmes=config.pos_bmes == 'off',
            max_text_len=config.max_sentence_length,
            max_span_len=config.max_span_length,
            sort_by_length=True), cache=config.load_from_cache)  # type: ConllDataSet
    dev_set = auto_create(
        "dev_set.{}".format(config.use_data_set),
        lambda: ConllDataSet(
            data_path=used_data_set[1],
            char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx,
            pos2idx=pos2idx, ner2idx=ner2idx, label2idx=label2idx,
            max_text_len=config.max_sentence_length,
            ignore_pos_bmes=config.pos_bmes == 'off',
            sort_by_length=False), cache=config.load_from_cache)  # type: ConllDataSet
    test_set = auto_create(
        "test_set.{}".format(config.use_data_set),
        lambda: ConllDataSet(
            data_path=used_data_set[2],
            char2idx=char2idx, bichar2idx=bichar2idx, seg2idx=seg2idx,
            pos2idx=pos2idx, ner2idx=ner2idx, label2idx=label2idx,
            max_text_len=config.max_sentence_length,
            ignore_pos_bmes=config.pos_bmes == 'off',
            sort_by_length=False), cache=config.load_from_cache)  # type: ConllDataSet
    longest_span_len = max(train_set.longest_span_len, dev_set.longest_span_len)
    longest_text_len = max(train_set.longest_text_len, dev_set.longest_text_len)

    luban7 = Luban7(char2idx=char2idx,
                    bichar2idx=bichar2idx,
                    seg2idx=seg2idx,
                    pos2idx=pos2idx,
                    ner2idx=ner2idx,
                    label2idx=label2idx,
                    longest_text_len=longest_text_len).to(device)
    if config.opt == 'adam':
        opt = torch.optim.Adam(luban7.parameters(), lr=0.001, weight_decay=config.weight_decay)
    elif config.opt == 'sparseadam':
        opt = torch.optim.SparseAdam(luban7.parameters(), lr=0.001)
    else:
        raise Exception
    manager = ModelManager(luban7, config.model_name, init_ckpt=config.model_ckpt) \
        if config.model_name != "off" else None
    # opt = torch.optim.SGD(luban7.parameters(), lr=0.1, momentum=0.9)

    epoch_id = -1
    while True:
        epoch_id += 1
        if epoch_id == config.epoch_max:
            break

        """
        Training
        """
        crf_evaluator = CRFEvaluator(idx2tag=idx2ner)
        if config.train_on:
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

                crf_loss = luban7.crf_nll(batch_data)
                preds = luban7.crf_decode(batch_data)
                crf_accu = crf_evaluator.eval(preds, group_fields(batch_data, "ners"))[2]

                score, span_ys = luban7.get_span_score_tags(batch_data)
                luban_loss = focal_loss(inputs=score,
                                        targets=torch.tensor(span_ys).to(device),
                                        gamma=config.focal_gamma)
                score_probs = F.softmax(score, dim=1)
                luban_accu = accuracy(score_probs.detach().cpu().numpy(), span_ys)

                loss = config.crf * crf_loss + (1 - config.crf) * luban_loss

                progress.update(len(batch_data))
                log("".join([
                    "[{}: {}/{}] ".format(epoch_id, progress.complete_num, train_set.size),
                    "b: {:.4f} / c:{:.4f} / r: {:.4f} "
                        .format(progress.batch_time, progress.cost_time, progress.rest_time),
                    "crf loss: {:.4f} ".format(crf_loss.item()),
                    "accuracy: {:.4f}".format(crf_accu)])
                )
                log("".join([
                    "[{}: {}/{}] ".format(epoch_id, progress.complete_num, train_set.size),
                    "b: {:.4f} / c:{:.4f} / r: {:.4f} "
                        .format(progress.batch_time, progress.cost_time, progress.rest_time),
                    "loss: {:.4f} ".format(luban_loss.item()),
                    "accuracy: {:.4f}".format(luban_accu)])
                )

                opt.zero_grad()
                loss.backward()
                if torch.isnan(luban7.embeds.char_embeds.weight.grad.sum()):
                    pdb.set_trace()
                clip_grad_norm_(luban7.parameters(), 5)
                opt.step()

            log("<<< epoch {} train".format(epoch_id))

        if isinstance(manager, ModelManager):
            manager.save()

        """
        Development
        """
        crf_evaluator = CRFEvaluator(idx2tag=idx2ner)
        luban_evaluator = LubanEvaluator()
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

                    # CRF
                    results = luban7.crf_decode(batch_data)
                    crf_evaluator.eval(results, group_fields(batch_data, "ners"))

                    # Luban
                    score, span_ys = luban7.get_span_score_tags(batch_data)
                    score_probs = F.softmax(score, dim=1)

                    pred = cast_list(torch.argmax(score, 1))

                    offset = 0
                    for bid in range(len(text_lens)):
                        log_to_buffer("[{:>4}] {}".format(
                            progress.complete_num + bid,
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
                        luban_evaluator.decode(luban_spans)
                        offset += len(enum_spans)
                    log_flush_buffer()

                    progress.update(len(batch_data))

                log("CRF: precision {:.4f}, recall {:.4f}, f1 {:.4f}".format(*crf_evaluator.prf))
                log("Luban: precision {:.4f}, recall {:.4f}, f1 {:.4f}".format(*luban_evaluator.prf))

                log("<<< epoch {} validation on {}".format(epoch_id, set_name))


if __name__ == '__main__':
    main()
