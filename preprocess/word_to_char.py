import re

in_file_path = "test.word.bmes"
out_file_path = "test.char.bmes"
out_file = open(out_file_path, "w")


# CHAR SEG POS NER

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


sentences = load_sentences(in_file_path)

for sentence in sentences:
    char_results = []
    for token in sentence:
        if len(token[0]) == 1:
            char_results.append((token[0], "S", "S-" + token[1], token[2]))
        if len(token[0]) > 1:
            if "B-" in token[2]:
                for i in range(len(token[0])):
                    if i == 0:
                        char_results.append((token[0][i], "B", "B-" + token[1], "B-" + token[2][2:]))
                    elif i == len(token[0]) - 1:
                        char_results.append((token[0][i], "E", "E-" + token[1], "M-" + token[2][2:]))
                    else:
                        char_results.append((token[0][i], "M", "M-" + token[1], "M-" + token[2][2:]))
            elif "M-" in token[2]:
                for i in range(len(token[0])):
                    if i == 0:
                        char_results.append((token[0][i], "B", "B-" + token[1], "M-" + token[2][2:]))
                    elif i == len(token[0]) - 1:
                        char_results.append((token[0][i], "E", "E-" + token[1], "M-" + token[2][2:]))
                    else:
                        char_results.append((token[0][i], "M", "M-" + token[1], "M-" + token[2][2:]))
            elif "E-" in token[2]:
                for i in range(len(token[0])):
                    if i == 0:
                        char_results.append((token[0][i], "B", "B-" + token[1], "M-" + token[2][2:]))
                    elif i == len(token[0]) - 1:
                        char_results.append((token[0][i], "E", "E-" + token[1], "E-" + token[2][2:]))
                    else:
                        char_results.append((token[0][i], "M", "M-" + token[1], "E-" + token[2][2:]))
            elif "S-" in token[2]:
                for i in range(len(token[0])):
                    if i == 0:
                        char_results.append((token[0][i], "B", "B-" + token[1], "B-" + token[2][2:]))
                    elif i == len(token[0]) - 1:
                        char_results.append((token[0][i], "E", "E-" + token[1], "E-" + token[2][2:]))
                    else:
                        char_results.append((token[0][i], "M", "M-" + token[1], "M-" + token[2][2:]))
            else:
                for i in range(len(token[0])):
                    if i == 0:
                        char_results.append((token[0][i], "B", "B-" + token[1], token[2]))
                    elif i == len(token[0]) - 1:
                        char_results.append((token[0][i], "E", "E-" + token[1], token[2]))
                    else:
                        char_results.append((token[0][i], "M", "M-" + token[1], token[2]))
    for ele in char_results:
        print(ele[0], ele[1], ele[2], ele[3], file=out_file)
    print(file=out_file)
    # exit()
