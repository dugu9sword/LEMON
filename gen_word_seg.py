from buff import log, log_config

log_config("small_train.goldseg.bmes", default_target="cf")

if __name__ == '__main__':
    file_name = "dataset/OntoNotes4/small_train.word.bmes"
    file = open(file_name)
    while True:
        line = file.readline()
        if line == "":
            break
        if line == "\n":
            log()
        else:
            word = line.split(" ")[0]
            word_len = len(word)
            for i, char in enumerate(word):
                if word_len == 1:
                    log(char, "S")
                else:
                    if i == 0:
                        log(char, "B")
                    elif i == word_len - 1:
                        log(char, "E")
                    else:
                        log(char, "M")
