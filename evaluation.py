from typing import List, Dict, NamedTuple, Set
import re
from buff import log

CRFSpan = NamedTuple("CRFSpan", [("bid", int), ("eid", int), ("label", str)])

EPS = 1e-10


def _prf(corr_num, pred_num, gold_num):
    precision = corr_num / (pred_num + EPS)
    recall = corr_num / (gold_num + EPS)
    f1 = 2 * precision * recall / (precision + recall + EPS)
    return precision, recall, f1


class CRFEvaluator:
    def __init__(self, idx2tag: Dict[int, str]):
        self.corr_num = 0.0
        self.gold_num = 0.0
        self.pred_num = 0.0
        self.idx2tag = idx2tag

    def eval(self,
             preds: List[List[int]],
             golds: List[List[int]]):
        _gold_num = 0.0
        _pred_num = 0.0
        _corr_num = 0.0
        for sid in range(len(preds)):
            pred_spans = self.to_span(preds[sid])
            gold_spans = self.to_span(golds[sid])
            _gold_num += len(gold_spans)
            _pred_num += len(pred_spans)
            _corr_num += len(gold_spans.intersection(pred_spans))
        self.gold_num += _gold_num
        self.pred_num += _pred_num
        self.corr_num += _corr_num
        return _prf(_corr_num, _pred_num, _gold_num)

    @property
    def prf(self):
        return _prf(self.corr_num, self.pred_num, self.gold_num)

    def to_span(self, tag_ids: List[int]) -> Set[CRFSpan]:
        tags = list(map(lambda x: self.idx2tag[x], tag_ids))
        spans = set()
        idx = 0
        while idx < len(tags):
            tag = tags[idx]
            if re.match("^S", tag):
                if len(tag) > 1:
                    label = tag[2:]
                else:
                    label = 'LABEL'
                spans.add(CRFSpan(bid=idx,
                                  eid=idx,
                                  label=label))
                idx += 1
                continue
            if re.match("^B", tag):
                bid = idx
                while idx < len(tags):
                    tag = tags[idx]
                    if re.match("^E", tag):
                        break
                    idx += 1
                if len(tag) > 1:
                    label = tag[2:]
                else:
                    label = 'LABEL'
                spans.add(CRFSpan(bid=bid,
                                  eid=idx,
                                  label=label))
            idx += 1
        return spans


LubanSpan = NamedTuple("SpanPred", [("bid", int),
                                    ("eid", int),
                                    ("lid", int),
                                    ("pred_prob", float),
                                    ("gold_prob", float),
                                    ("pred_label", str),
                                    ("gold_label", str),
                                    ("fragment", str)])


def luban_span_to_str(luban_span: LubanSpan):
    return "{:>3}~{:<3}\t{:1}\t{:.4f}/{:.4f}\t{:>5}/{:<5}\t{}".format(
        luban_span.bid, luban_span.eid, luban_span.lid,
        luban_span.pred_prob, luban_span.gold_prob,
        luban_span.pred_label, luban_span.gold_label,
        luban_span.fragment,
    )


class LubanEvaluator:
    def __init__(self):
        self.corr_num = 0.
        self.gold_num = 0.
        self.pred_num = 0.
        self.TP = {"PER": 0., "GPE": 0., "LOC": 0., "ORG": 0., "NONE": 0.}  # in gold, in pred
        self.FP = {"PER": 0., "GPE": 0., "LOC": 0., "ORG": 0., "NONE": 0.}  # not in gold, in pred
        self.FN = {"PER": 0., "GPE": 0., "LOC": 0., "ORG": 0., "NONE": 0.}  # in gold, not in pred

    def decode(self,
               results: List[LubanSpan],
               threshold=0,
               verbose=False):
        keep_flags = [results[i].pred_label != 'NONE' for i in range(len(results))]

        # By threshold
        for i, span_pred in enumerate(results):
            if span_pred.pred_prob < threshold:
                keep_flags[i] = False

        # Bigger is better
        for i in range(len(results) - 1):
            if results[i + 1].bid == results[i].bid:
                keep_flags[i] = False

        # Overlapping
        while True:
            no_conflict = True
            for i in range(len(results)):
                if keep_flags[i]:
                    next_id = i + 1
                    while not next_id == len(results) and not keep_flags[next_id]:
                        next_id += 1
                    if next_id == len(results):
                        continue
                    if results[next_id].bid <= results[i].eid:
                        if results[next_id].pred_prob > results[i].pred_prob:
                            keep_flags[i] = False
                        else:
                            keep_flags[next_id] = False
                        no_conflict = False
            if no_conflict:
                break

        if verbose:
            for i in range(len(results)):
                if keep_flags[i] or results[i].gold_label != 'NONE':
                    log("{} {:>4}/{:4} {}".format(
                        "+" if keep_flags[i] else "-",
                        results[i].pred_label, results[i].gold_label, results[i].fragment))

        for i, span_pred in enumerate(results):
            if keep_flags[i] and span_pred.gold_label == span_pred.pred_label:
                self.TP[span_pred.gold_label] += 1
                self.corr_num += 1
                self.pred_num += 1
                self.gold_num += 1
            if keep_flags[i] and span_pred.gold_label != span_pred.pred_label:
                self.FP[span_pred.pred_label] += 1
                self.FN[span_pred.gold_label] += 1
                self.pred_num += 1
                if span_pred.gold_label != 'NONE':
                    self.gold_num += 1
            if not keep_flags[i] and span_pred.gold_label != 'NONE':
                self.FN[span_pred.gold_label] += 1
                self.gold_num += 1

    @property
    def prf(self, verbose=True):
        if verbose:
            log("TP", self.TP)
            log("FP", self.FP)
            log("FN", self.FN)
        mi_pre = (0, 0)
        mi_rec = (0, 0)
        ma_pre = 0.
        ma_rec = 0.
        for entype in ['GPE', 'LOC', 'ORG', 'PER']:
            pre = self.TP[entype] / (self.TP[entype] + self.FP[entype] + 1e-10)
            rec = self.TP[entype] / (self.TP[entype] + self.FN[entype] + 1e-10)
            ma_pre += pre
            ma_rec += rec
            mi_pre = (mi_pre[0] + self.TP[entype], mi_pre[1] + self.TP[entype] + self.FP[entype])
            mi_rec = (mi_rec[0] + self.TP[entype], mi_rec[1] + self.TP[entype] + self.FN[entype])
            f1 = 2 * pre * rec / (pre + rec + 1e-10)
            if verbose:
                log("{} pre: {:.4f} rec: {:.4f} f1:  {:.4f}".format(
                    entype, pre, rec, f1
                ))
        mi_pre = mi_pre[0] / (mi_pre[1] + 1e-10)
        mi_rec = mi_rec[0] / (mi_rec[1] + 1e-10)
        mi_f1 = 2 * mi_pre * mi_rec / (mi_pre + mi_rec + 1e-10)
        ma_pre /= 4
        ma_rec /= 4
        ma_f1 = 2 * ma_pre * ma_rec / (ma_pre + ma_rec + 1e-10)
        if verbose:
            log("micro pre: {:.4f} rec: {:.4f} f1:  {:.4f}".format(mi_pre, mi_rec, mi_f1))
            log("macro pre: {:.4f} rec: {:.4f} f1:  {:.4f}".format(ma_pre, ma_rec, ma_f1))

            # log("ignore-class pre: {:.4f} rec: {:.4f} f1:  {:.4f}".format(
            #     self.corr_num / (self.pred_num + 1e-10),
            #     self.corr_num / (self.gold_num + 1e-10),
            #     2 / (self.pred_num / (self.corr_num + 1e-10) + self.gold_num / (self.corr_num + 1e-10))
            # ))
        return mi_pre, mi_rec, mi_f1

# crf = CRFEvaluator(idx2tag={
#     0: 'O', 1: 'B-GPE', 2: 'M-GPE', 3: 'E-GPE', 4: 'S-GPE'
# })
#
# crf.eval(preds=[[0, 0, 4, 1, 1, 2, 3, 0]],
#          golds=[[0, 0, 4, 0, 1, 2, 3, 0]])
