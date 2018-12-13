import re
from buff import log, log_config

def load_sentences(file_path, sep=r"\s+"):
    ret = []
    sentence = []
    for line in open(file_path, encoding='utf8'):
        line = line.strip("\n")
        if line == "":
            if not sentence == []:
                ret.append(sentence)
                sentence = []
        else:
            sentence.append(re.split(sep, line))
    return ret


if __name__ == '__main__':
    log_config("dev.sentences", 'cf')
    sentences = load_sentences("dataset/OntoNotes4/dev.char.bmes")
    for sid, sen in enumerate(sentences):
        log("[{:>5}] {}".format(sid, "".join([ele[0] for ele in sen])))
