import re


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


file = "../dataset/weibo/test.ner.bmes"
out_file = "../dataset/weibo/test.ner.bmes.noseg"

sentences = load_sentences(file)

if out_file is None:
    out = None
else:
    out = open(out_file, "w", encoding='utf8')

for s in sentences:
    for ch, tag in s:
        print(ch[:-1], tag, file=out)
    print(file=out)
