import os
from colorama import Fore, Back
from typing import List

__log_path__ = "logs"
globals()["__default_target__"] = 'c'


def log_config(filename,
               default_target,
               log_path=__log_path__,
               append=False,
               ):
    if not os.path.exists(log_path):
        os.makedirs(log_path, exist_ok=True)
    logger = open("{}/{}".format(log_path, filename),
                  "a" if append else "w")
    globals()["__logger__"] = logger
    globals()["__default_target__"] = default_target


def log(*info, target=None):
    if target is None:
        target = globals()["__default_target__"]
    assert target in ['c', 'f', 'cf', 'fc']
    if len(info) == 1:
        info_str = str(info[0])
    else:
        info = list(map(str, info))
        info_str = " ".join(info)
    if 'c' in target:
        print(info_str)
    if 'f' in target:
        logger = globals()["__logger__"]
        logger.write("{}\n".format(info_str))
        logger.flush()


log_buffer = []  # type:List


def log_to_buffer(*info):
    for ele in info:
        log_buffer.append(ele)


def log_flush_buffer(target=None):
    log("\n".join(log_buffer), target=target)
    log_buffer.clear()


class Color(object):
    @staticmethod
    def red(s):
        return Fore.RED + s + Fore.RESET

    @staticmethod
    def green(s):
        return Fore.GREEN + s + Fore.RESET

    @staticmethod
    def yellow(s):
        return Fore.YELLOW + s + Fore.RESET

    @staticmethod
    def blue(s):
        return Fore.BLUE + s + Fore.RESET

    @staticmethod
    def magenta(s):
        return Fore.MAGENTA + s + Fore.RESET

    @staticmethod
    def cyan(s):
        return Fore.CYAN + s + Fore.RESET

    @staticmethod
    def white(s):
        return Fore.WHITE + s + Fore.RESET

    @staticmethod
    def white_green(s):
        return Fore.WHITE + Back.GREEN + s + Fore.RESET + Back.RESET
