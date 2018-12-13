vocab_path = "dataset/OntoNotes4/vocab.3.2/bichar.vocab"
emb_path = "word2vec/lattice_lstm/gigaword_chn.all.a2b.bi.ite50.vec"

embs = set()
for line in open(emb_path, encoding="utf8"):
    word = line.split(" ")[0]
    # if "<s>" in word:
    #     print(word)
    embs.add(word)

vocab = set()
for line in open(vocab_path, encoding="utf8"):
    vocab.add(line.split(" ")[0])

in_num = 0
oov_num = 0
for ele in vocab:
    if ele in embs:
        in_num += 1
    else:
        oov_num += 1
        # print(ele)
print(in_num)
print(oov_num)


