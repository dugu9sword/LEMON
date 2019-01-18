import os
import re
from collections import OrderedDict
from typing import List
import pdb


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
# root_path = "/mnt/c/Users/zhouyi/Desktop/ontonotes-release-4.0/data/files/data/chinese/annotations/"

files = []
for path, d, filelist in os.walk(root_path):
    # print(d)
    for filename in filelist:
        files.append(os.path.join(path, filename))

tag_files = OrderedDict()
ner_files = OrderedDict()

print(files)

for file in files:
    found = re.search(r".*/([^/]*)\.([a-z]*)", file)
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


def extract_ner(raw_str, keep_labels=("PERSON", "GPE", "ORG", "LOC")):
    ret = []
    found = re.findall(r'(?:<ENAMEX TYPE="[^"]*"[^>]*>[^<]+</ENAMEX>|[^\s]+)', raw_str)
    for ele in found:
        entity = re.search(r'<ENAMEX TYPE="([^"]*)"[^>]*>([^<]+)</ENAMEX>', ele)
        if entity:
            entities = re.split(r"\s+", entity.group(2))
            if entity.group(1) not in keep_labels:
                for ent in entities:
                    ret.append((ent, "O"))
            else:
                if len(entities) == 1:
                    ret.append((entities[0], 'S-{}'.format(entity.group(1))))
                else:
                    ret.append((entities[0], 'B-{}'.format(entity.group(1))))
                    if len(entities) >= 3:
                        for i in range(1, len(entities) - 1):
                            ret.append((entities[i], 'M-{}'.format(entity.group(1))))
                    ret.append((entities[-1], 'E-{}'.format(entity.group(1))))
        else:
            ret.append((ele, "O"))
    return ret


from buff import log, log_config

train_file = open("train.word.bmes", "w", encoding="utf8")
dev_file = open("dev.word.bmes", "w", encoding="utf8")
test_file = open("test.word.bmes", "w", encoding="utf8")

train_num = 0
dev_num = 0
test_num = 0
for idx in ner_files:
    net_sentence_lst = open(ner_files[idx], encoding='utf8').readlines()[1:-1]
    tag_word_lst_lst = load_sentences(tag_files[idx])  # type: List[List[str]]

    for ner_line_idx in range(len(net_sentence_lst)):
        ner_sentence = net_sentence_lst[ner_line_idx].strip("\n")  # type:str
        tag_word_lst = tag_word_lst_lst[ner_line_idx]  # type:List[str]
        ner_pairs = extract_ner(ner_sentence)
        print(ner_sentence)
        print(tag_word_lst)
        word_results = []
        for ele in ner_pairs:
            tag_word_idx = 0
            try:
                while True:
                    # print(tag_line[tag_line_no])
                    if ele[0] in ["?", ".", "(", ")", "$"]:
                        found_tag = re.search(r'\(([^\s]+)\s+{}\)'.format("\\" + ele[0]),
                                              tag_word_lst[tag_word_idx])
                    else:
                        found_tag = re.search(r'\(([^\s]+)\s+{}\)'.format(ele[0]),
                                              tag_word_lst[tag_word_idx])
                    tag_word_idx += 1
                    if found_tag:
                        break
            except:
                pdb.set_trace()
            word_results.append((ele[0], found_tag.group(1), ele[1][:5]))

        if "chtb" not in idx:
            fd = train_file
            train_num += len(tag_word_lst_lst)
        else:
            if re.match(".*[13579]$", idx):
                fd = dev_file
                dev_num += len(tag_word_lst_lst)
            else:
                fd = test_file
                test_num += len(tag_word_lst_lst)
        for ele in word_results:
            line = "{} {} {}\n".format(ele[0], ele[1], ele[2])
            # line = "{} {}\n".format(ele[0], ele[2])
            fd.write(line)
        fd.write("\n")
        # print(ele[0], found_tag.group(1))
        # exit()

print(train_num, dev_num, test_num)
