from dataset import ConllDataSet

# for line in open("word2vec/lattice_lstm/ctb.50d.vec"):
#     split = line.split(" ")

from buff import *
import torch
import pandas
import time


def load_embedding(word_dict: Dict[str, int],
                   word2vec_path,
                   norm=True,
                   cached_name=None):
    cache = "{}{}".format(cached_name, ".norm" if norm else "")
    if cached_name and exist_var(cache):
        log("Load vocab from cache {}".format(cache))
        pre_embedding = load_var(cache)
    else:
        log("Load vocab from {}".format(word2vec_path))
        pre_embedding = np.random.normal(0, 1, len(word_dict))
        word2vec_file = open(word2vec_path, errors='ignore')
        # x = 0
        found = 0
        for line in word2vec_file.readlines():
            # x += 1
            # log("Process line {} in file {}".format(x, word2vec_path))
            split = re.split(r"\s+", line.strip())
            # for word2vec, the first line is meta info: (NUMBER, SIZE)
            if len(split) < 10:
                continue
            word = split[0]
            if word in word_dict:
                found += 1
                pre_embedding[word_dict[word]] = np.array(list(map(float, split[1:])))
        log("Pre_train match case: {:.4f}".format(found / len(word_dict)))
        if norm:
            pre_embedding = pre_embedding / np.std(pre_embedding)
        if cached_name:
            save_var(pre_embedding, cache)


def load_embedding_v2(word2vec_path,
                      norm=True,
                      cached_name=None):
    cache = "{}{}".format(cached_name, ".norm" if norm else "")
    if cached_name and exist_var(cache):
        log("Load vocab from cache {}".format(cache))
        word_dict, embeddings = load_var(cache)
    else:
        log("Load vocab from {}".format(word2vec_path))
        csv = pandas.read_csv(word2vec_path, sep="\\s+")
        words = csv.values[:, 0]
        embeddings = csv.values[:, 1:]
        word_dict = {}
        for idx, word in enumerate(words):
            word_dict[word] = idx
        print("Embedding num: {} dim: {} mean: {:.3f} std: {:.3f}".format(
            *embeddings.shape, embeddings.mean(), embeddings.std()
        ))
        if norm:
            embeddings = embeddings / embeddings.std()
        if cached_name:
            save_var(embeddings, cache)
    return word_dict, embeddings


load_embedding_v2("word2vec/lattice_lstm/ctb.50d.vec")
show_mem()
