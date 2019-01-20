import re
from typing import NamedTuple, List
from buff import Color, log, log_config
import os
import numpy as np

SpanPred = NamedTuple("SpanPred", [("bid", int),
                                   ("eid", int),
                                   ("lid", int),
                                   ("pred_prob", float),
                                   ("gold_prob", float),
                                   ("pred_label", str),
                                   ("gold_label", str),
                                   ("fragment", str)])


class Decoder:
    def __init__(self):
        self.corr_num = 0.
        self.gold_num = 0.
        self.pred_num = 0.
        self.TP = {"PER": 0., "GPE": 0., "LOC": 0., "ORG": 0., "NONE": 0.}  # in gold, in pred
        self.FP = {"PER": 0., "GPE": 0., "LOC": 0., "ORG": 0., "NONE": 0.}  # not in gold, in pred
        self.FN = {"PER": 0., "GPE": 0., "LOC": 0., "ORG": 0., "NONE": 0.}  # in gold, not in pred

    def decode(self,
               results: List[SpanPred],
               threshold,
               verbose):
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

    def evaluate(self, verbose=True):
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


def decode_log(file_path="lstm.json.logs/last.task-4.txt",
               threshold=-4,
               verbose=True,
               epoch_id=29,
               valid_set="dev_set"):
    file = open(file_path)

    decoder = Decoder()

    results: List[SpanPred] = []
    flag = False
    while True:
        line = file.readline()
        if line == '':
            break
        line = line.strip("\n")
        if line == ">>> epoch {} validation on {}".format(epoch_id, valid_set):
            flag = True
        if line == "<<< epoch {} validation on {}".format(epoch_id, valid_set):
            break
        if flag:
            if re.match(r"\[[^\d]*\d+\].*", line):
                if len(results) != 0:
                    decoder.decode(results, threshold, verbose)
                results = []
                if verbose:
                    log(line)
            found = re.search(r"\s+(\d+)~(\d+)\s+(\d+)\s+(\d*\.\d*)/(\d*\.\d*)\s*([A-Z]+)/([A-Z]+)\s*([^\s]*)", line)
            if found:
                results.append(SpanPred(bid=int(found.group(1)),
                                        eid=int(found.group(2)),
                                        lid=int(found.group(3)),
                                        pred_prob=float(found.group(4)),
                                        gold_prob=float(found.group(5)),
                                        pred_label=found.group(6),
                                        gold_label=found.group(7),
                                        fragment=found.group(8)))
                # print(results)
    return decoder.evaluate(verbose)


if __name__ == '__main__':
    log_config("verbose.txt", "cf")
    # folder = "/home/zhouyi/Desktop/pretrain/pretrain.json.logs/"
    # for file_idx in range(len(list(filter(lambda x: "last" in x, os.listdir(folder))))):
    #     try:
    #         mi_pre, mi_rec, mi_f1 = decode_log(file_path="{}/last.task-{}.txt".format(folder, file_idx),
    #                                            threshold=-4,
    #                                            verbose=False)
    #     except:
    #         mi_pre, mi_rec, mi_f1 = -1, -1, -1
    #     print("{} {} {} {}".format(file_idx, mi_pre, mi_rec, mi_f1))
    # for file_id in range(0, 8):
    #     try:
    #         print(file_id,
    #               decode_log(file_path="/home/zhouyi/Desktop/task.json.logs/task-{}.txt".format(file_id),
    #                          threshold=-100,
    #                          verbose=False,
    #                          epoch_id=29,
    #                          valid_set="dev_set"))
    #     except:
    #         pass

    # np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
    # file_num = 16
    # results = np.zeros((file_num, 40), dtype=float)
    # for file_id in range(0, file_num):
    #     print(file_id)
    #     for epoch_id in range(0, 40):
    #         try:
    #             f1 = decode_log(
    #                 file_path="/home/zhouyi/Desktop/attention.json.logs/task-{}.txt".format(file_id),
    #                 threshold=0,
    #                 verbose=False,
    #                 epoch_id=epoch_id,
    #                 valid_set="dev_set")[2]
    #             print(file_id, epoch_id, f1)
    #             results[file_id, epoch_id] = f1
    #         except:
    #             pass
    # results = results.T
    # for i in range(40):
    #     for j in range(file_num):
    #         print("{:0.4f}".format(results[i, j]), sep=", ")
    #     print()

    # for i in range(0, 40):
    #     print(decode_log(file_path="/home/zhouyi/Desktop/attention.json.logs/task-0.txt",
    #                      threshold=-8,
    #                      verbose=False,
    #                      epoch_id=i,
    #                      valid_set="dev_set")[2])

    print(decode_log(file_path="logs/main.txt",
                     threshold=-8,
                     verbose=False,
                     epoch_id=0,
                     valid_set="dev_set")[2])
