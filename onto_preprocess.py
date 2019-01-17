import os
import re
from collections import OrderedDict


def load_sentences(file_path):
    ret = []
    sentence = []
    for line in open(file_path, encoding='utf8'):
        line = line.strip("\n")
        if line == "":
            if not sentence == []:
                ret.append(sentence)
                sentence = []
        else:
            sentence.append(line)
    return ret


root_path = "/home/zhouyi/hdd/NER/ontonotes-release-4.0/data/files/data/chinese/annotations/"

files = []
for path, d, filelist in os.walk(root_path):
    # print(d)
    for filename in filelist:
        files.append(os.path.join(path, filename))

tag_files = OrderedDict()
ner_files = OrderedDict()

for file in files:
    found = re.search(".*/([^/]*)\.([a-z]*)", file)
    if found:
        if found.group(2) == 'name':
            ner_files[found.group(1)] = file
        if found.group(2) == 'parse':
            tag_files[found.group(1)] = file

print("检查所有的命名实体文件都有对应的句法文件：")
for ner_idx in ner_files.keys():
    if ner_idx not in tag_files:
        print(ner_idx)
print("检查完毕。")

ner_files = OrderedDict(sorted(ner_files.items(),
                               key=lambda item: item[0]))

total=0
for idx in ner_files:
    print(ner_files[idx])
    ner_lines = open(ner_files[idx], encoding='utf8').readlines()
    print(tag_files[idx])
    tag_lines = load_sentences(tag_files[idx])
    print(len(tag_lines))
    #print(tag_lines)
    print(len(ner_lines))

    total+=len(tag_lines)

print(total)
