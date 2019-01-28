import logging
import os
import time
from contextlib import contextmanager
import numpy as np
import re
import pickle
import random
import argparse
from typing import List, Dict, NamedTuple, Union
from colorama import Fore, Back
import psutil
from sklearn.metrics import precision_recall_fscore_support as sklearn_prf

__saved_path__ = "saved/vars"

arg_required = object()
arg_optional = object()
arg_place_holder = object()


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)


def show_mem():
    top = psutil.Process(os.getpid())
    info = top.memory_full_info()
    memory = info.uss / 1024. / 1024.
    print('Memory: {:.2f} MB'.format(memory))


def set_saved_path(path):
    global __saved_path__
    __saved_path__ = path


def save_var(variable, name, path=None):
    if path is None:
        path = __saved_path__
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    pickle.dump(variable, open("{}/{}.pkl".format(path, name), "wb"))


def load_var(name, path=None):
    if path is None:
        path = __saved_path__
    return pickle.load(open("{}/{}.pkl".format(path, name), "rb"))


def exist_var(name, path=None):
    if path is None:
        path = __saved_path__
    return os.path.exists("{}/{}.pkl".format(path, name))


def auto_create(name, func, cache=False, path=__saved_path__):
    if cache and exist_var(name, path):
        obj = load_var(name, path)
    else:
        obj = func()
        save_var(obj, name, path)
    return obj


@contextmanager
def time_record():
    start = time.time()
    yield
    end = time.time()
    print("cost {:.3} seconds".format(end - start))


class ProgressManager:
    def __init__(self, total):
        self.__start = time.time()
        self.__prev_prev = time.time()
        self.__prev = time.time()
        self.__total = total
        self.__complete = 0

    def update(self, batch_num):
        self.__complete += batch_num
        self.__prev_prev = self.__prev
        self.__prev = time.time()

    @property
    def batch_time(self):
        return self.__prev - self.__prev_prev

    @property
    def cost_time(self):
        return self.__prev - self.__start

    @property
    def rest_time(self):
        return self.cost_time / self.__complete * (self.__total - self.__complete)

    @property
    def complete_num(self):
        return self.__complete

    @property
    def total_num(self):
        return self.__total


class DataSet:
    def __init__(self):
        self.data = []
        self.__next = 0

    def next_batch(self, batch_size, fill_batch=True):
        if self.__next + batch_size > len(self.data):
            if fill_batch:
                ret = self.data[self.size - batch_size:self.size]
            else:
                ret = self.data[self.__next:self.size]
            self.__next = self.size
        else:
            ret = self.data[self.__next:self.__next + batch_size]
            self.__next += batch_size
        return ret

    @property
    def size(self):
        return len(self.data)

    @property
    def finished(self):
        return self.__next == self.size

    def reset(self, shuffle=True):
        self.__next = 0
        if shuffle:
            random.shuffle(self.data)


class ArgParser:
    def __init__(self):
        self.ap = argparse.ArgumentParser()

    def request(self, key, value):
        self.ap.add_argument('-{}'.format(key),
                             action='store',
                             default=value,
                             type=type(value),
                             dest=str(key))

    def parse(self):
        return self.ap.parse_args()


def hit(scores: List[List], gold: List, k: int):
    corr = 0
    total = len(gold)
    for score, label in zip(scores, gold):
        if label in list(reversed(np.argsort(score)))[:k]:
            corr += 1
    return corr / total


def get_prf(pred: List, gold: List) -> (List, List, List):
    precision, recall, f1, _ = sklearn_prf(gold, pred, beta=1, average=None)
    return precision.tolist(), recall.tolist(), f1.tolist()


def show_prf(pred: List, gold: List, classes):
    precision, recall, f1, _ = sklearn_prf(gold, pred, beta=1, labels=classes)
    head = "{:4}|{:15}|{:10}|{:10}|{:10}"
    content = "{:4}|{:15}|{:10f}|{:10f}|{:10f}"
    print(Color.cyan(head.format("ID", "Class", "Precision", "Recall", "F1")))
    for i in range(len(classes)):
        print(Color.white(content.format(i, classes[i], precision[i], recall[i], f1[i])))


def score2rank(scores) -> list:
    return np.argmax(scores, 1).tolist()


def accuracy(scores: List[List], gold: List):
    return hit(scores, gold, 1)


class Color(object):
    @staticmethod
    def red(s):
        return Fore.RED + str(s) + Fore.RESET

    @staticmethod
    def green(s):
        return Fore.GREEN + str(s) + Fore.RESET

    @staticmethod
    def yellow(s):
        return Fore.YELLOW + str(s) + Fore.RESET

    @staticmethod
    def blue(s):
        return Fore.BLUE + str(s) + Fore.RESET

    @staticmethod
    def magenta(s):
        return Fore.MAGENTA + str(s) + Fore.RESET

    @staticmethod
    def cyan(s):
        return Fore.CYAN + str(s) + Fore.RESET

    @staticmethod
    def white(s):
        return Fore.WHITE + str(s) + Fore.RESET

    @staticmethod
    def white_green(s):
        return Fore.WHITE + Back.GREEN + str(s) + Fore.RESET + Back.RESET


class TrainingStopObserver:
    def __init__(self,
                 lower_is_better,
                 can_stop_val=None,
                 must_stop_val=None,
                 min_epoch=None,
                 max_epoch=None,
                 epoch_num=None
                 ):
        self.history_values = []
        self.history_infos = []
        self.min_epoch = min_epoch
        self.max_epoch = max_epoch
        self.epoch_num = epoch_num
        self.lower_is_better = lower_is_better
        self.can_stop_val = can_stop_val
        self.must_stop_val = must_stop_val

    def check_stop(self, value, info=None) -> bool:
        self.history_values.append(value)
        self.history_infos.append(info)
        if self.can_stop_val is not None:
            if self.lower_is_better and value > self.can_stop_val:
                return False
            if not self.lower_is_better and value < self.can_stop_val:
                return False
        if self.must_stop_val is not None:
            if self.lower_is_better and value < self.must_stop_val:
                return True
            if not self.lower_is_better and value > self.must_stop_val:
                return True
        if self.max_epoch is not None and len(self.history_values) > self.max_epoch:
            return True
        if self.min_epoch is not None and len(self.history_values) <= self.min_epoch:
            return False
        lower = value < np.mean(self.history_values[-(self.epoch_num + 1):-1])
        if self.lower_is_better:
            return False if lower else True
        else:
            return True if lower else False

    def select_best_point(self):
        if self.lower_is_better:
            chosen_id = int(np.argmin(self.history_values[self.min_epoch:]))
        else:
            chosen_id = int(np.argmax(self.history_values[self.min_epoch:]))
        return self.history_values[self.min_epoch + chosen_id], self.history_infos[self.min_epoch + chosen_id]


def cast_item(array):
    if isinstance(array, np.ndarray):
        array = array.tolist()
    while True:
        if isinstance(array, list):
            if len(array) != 1:
                raise Exception("More than one item!")
            array = array[0]
        else:
            break
    return array


def cast_list(array):
    if isinstance(array, list):
        return cast_list(np.array(array))
    if isinstance(array, np.ndarray):
        return array.squeeze().tolist()


class Collector:
    def __init__(self):
        self.has_key = False
        self.keys = None
        self.saved = None

    def collect(self, *args):
        # First called, init the collector and decide the key mode
        if self.saved is None:
            if Collector.__arg_has_key(*args):
                self.has_key = True
                self.keys = list(map(lambda x: x[0], args))
            self.saved = [[] for _ in range(len(args))]
        # Later called
        if Collector.__arg_has_key(*args) != self.has_key:
            raise Exception("you must always specify a key or not")
        for i in range(len(args)):
            if self.has_key:
                saved_id = self.keys.index(args[i][0])
                to_save = args[i][1]
            else:
                saved_id = i
                to_save = args[i]
            if isinstance(to_save, list):
                self.saved[saved_id].extend(to_save)
            else:
                self.saved[saved_id].append(to_save)

    @staticmethod
    def __arg_has_key(*args):
        # print("args is {}".format(args))
        has_key_num = 0
        for arg in args:
            if isinstance(arg, tuple) and len(arg) == 2 and isinstance(arg[0], str):
                has_key_num += 1
        if has_key_num == len(args):
            return True
        if has_key_num == 0:
            return False
        raise Exception("you must specify a key for all args or not")

    def collected(self, key=None):
        if key is None:
            if not self.has_key:
                if len(self.saved) == 1:
                    return self.saved[0]
                else:
                    return tuple(self.saved)
            else:
                raise Exception("you must specify a key")
        elif key is not None:
            if self.has_key:
                saved_id = self.keys.index(key)
                return self.saved[saved_id]
            else:
                raise Exception("you cannot specify a key")


def analyze_length_count(length_count: dict):
    sorted_count = sorted(length_count.items(), key=lambda kv: kv[0])
    print("Analyze length count:")
    print("\tTotal Count:", *sorted_count)
    pivots = [0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 1.01]
    agg_num = []
    tmp_num = 0
    for k, v in sorted_count:
        tmp_num += v
        agg_num.append(tmp_num)
    print("\tTotal num: ", tmp_num)
    agg_ratio = list(map(lambda x: x / tmp_num, agg_num))
    print("\tRatio: ")
    for pivot in pivots:
        idx = sum(list(map(lambda x: x < pivot, agg_ratio))) - 1
        print("\t\t{} : {}".format(pivot, "-" if idx == -1 else sorted_count[idx][0]))


def analyze_vocab_count(vocab_count: dict):
    pivots = [0, 1, 2, 3, 4, 5, 10, 20, 30]
    vocab_size = []
    count = []
    for pivot in pivots:
        tmp = list(filter(lambda pair: pair[1] > pivot, vocab_count.items()))
        vocab_size.append(len(tmp))
        count.append(sum(map(lambda pair: pair[1], tmp)))
    print("Analyze vocab count:")
    print("\tTotal vocab size {}, count: {}".format(vocab_size[0], count[0]))
    print("\tRatio: ")
    for cid in range(len(pivots)):
        print("\t\t> {}: vocab size {}/{:.3f}, count {}/{:.3f}".format(
            pivots[cid],
            vocab_size[cid], vocab_size[cid] / vocab_size[0],
            count[cid], count[cid] / count[0]
        ))


def group_fields(lst: List[object],
                 keys: Union[str, List[str]] = None,
                 indices: Union[int, List[int]] = None):
    assert keys is None or indices is None
    is_single = False
    if keys:
        if not isinstance(keys, list):
            keys = [keys]
            is_single = True
        indices = []
        for key in keys:
            obj_type = type(lst[0])
            idx = obj_type._fields.index(key)
            indices.append(idx)
    else:
        if not isinstance(indices, list):
            indices = [indices]
            is_single = True
    rets = []
    for idx in indices:
        rets.append(list(map(lambda item: item[idx], lst)))
    if is_single:
        return rets[0]
    else:
        return rets
