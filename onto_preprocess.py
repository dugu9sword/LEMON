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


# root_path = "/home/zhouyi/hdd/NER/ontonotes-release-4.0/data/files/data/chinese/annotations/"
root_path = "/mnt/c/Users/zhouyi/Desktop/ontonotes-release-4.0/data/files/data/chinese/annotations/"

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

log_config("train.txt", "cf")
# log_config("dev.txt", "cf")
# log_config("test.txt", "cf")

train_num = 0
dev_num = 0
test_num = 0
for idx in ner_files:
    # print(ner_files[idx])
    ner_lines = open(ner_files[idx], encoding='utf8').readlines()
    # print(tag_files[idx])
    tag_lines = load_sentences(tag_files[idx])
    # print(len(tag_lines))
    # print(tag_lines)
    # print(len(ner_lines))

    # print(ner_pairs)
    # if "暨" in line:
    #     break

    if "chtb" not in idx:
        train_num += len(tag_lines)
        for line_id in range(1, len(ner_lines) - 1):
            tag_line_no = 0

            ner_line = ner_lines[line_id]
            tag_line = tag_lines[line_id - 1]
            ner_pairs = extract_ner(ner_line)
            for ele in ner_pairs:
                # print(tag_line[tag_line_no])
                while True:
                    # print(tag_line[tag_line_no])
                    found_tag = re.search(r'\(([^\s]+)\s+{}\)'.format(ele[0]), tag_line[tag_line_no])
                    tag_line_no += 1
                    if found_tag:
                        break
                log("{} {} {}".format(ele[0], ele[1], found_tag.group(1)))
                # print(ele[0], found_tag.group(1))
            # exit()
            log()
    else:
        if re.match(".*[13579]$", idx):
            print(idx)
            dev_num += len(tag_lines)
        else:
            test_num += len(tag_lines)

print(train_num, dev_num, test_num)
