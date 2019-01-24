import re
from typing import NamedTuple, List
from buff import Color, log, log_config
import os
import numpy as np
from evaluation import LubanEvaluator, LubanSpan


def decode_log(file_path="lstm.json.logs/last.task-4.txt",
               threshold=-4,
               verbose=True,
               epoch_id=29,
               valid_set="dev_set"):
    file = open(file_path)

    decoder = LubanEvaluator()

    results: List[LubanSpan] = []
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
                results.append(LubanSpan(bid=int(found.group(1)),
                                         eid=int(found.group(2)),
                                         lid=int(found.group(3)),
                                         pred_prob=float(found.group(4)),
                                         gold_prob=float(found.group(5)),
                                         pred_label=found.group(6),
                                         gold_label=found.group(7),
                                         fragment=found.group(8)))
                # print(results)
    return decoder.prf(verbose)


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
    # file_num = 2
    # epoch_num = 30
    # results = np.zeros((file_num, epoch_num), dtype=float)
    # for file_id in range(0, file_num):
    #     print(file_id)
    #     for epoch_id in range(0, epoch_num):
    #         try:
    #             f1 = decode_log(
    #                 file_path="/home/zhouyi/Desktop/onlyctx.json.logs/task-{}.txt".format(file_id),
    #                 threshold=0,
    #                 verbose=False,
    #                 epoch_id=epoch_id,
    #                 valid_set="test_set")[2]
    #             print(file_id, epoch_id, f1)
    #             results[file_id, epoch_id] = f1
    #         except:
    #             pass
    # results = results.T
    # for i in range(epoch_num):
    #     for j in range(file_num):
    #         print("{:0.4f}".format(results[i, j]), sep=", ", end=", ")
    #     print()

    # for i in range(0, 40):
    #     print(decode_log(file_path="/home/zhouyi/Desktop/attention.json.logs/task-0.txt",
    #                      threshold=-8,
    #                      verbose=False,
    #                      epoch_id=i,
    #                      valid_set="dev_set")[2])

    for i in range(0, 30):
        print(decode_log(file_path="logs/main.txt",
                         threshold=0,
                         verbose=False,
                         epoch_id=i,
                         valid_set="dev_set")[2])
