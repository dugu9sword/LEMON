from .public import *
import torch
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from .logging import log

__model_path__ = "saved/models"


def cast_list(array):
    if isinstance(array, torch.Tensor):
        return cast_list(array.detach().cpu().numpy())
    if isinstance(array, list):
        return cast_list(np.array(array))
    if isinstance(array, np.ndarray):
        return array.squeeze().tolist()


def set_gpu_device(device_id):
    torch.cuda.set_device(device_id)


def gpu(*x):
    if torch.cuda.is_available():
        if len(x) == 1:
            return x[0].cuda()
        else:
            return map(lambda m: m.cuda(), x)
    else:
        if len(x) == 1:
            return x[0]
        else:
            return x


def load_model(model, saved_model_name, checkpoint=-1):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    if checkpoint == -1:
        for file in os.listdir(__model_path__):
            file = file[:-5]
            name = file.split('@')[0]
            ckpt = int(file.split('@')[1])
            if name == saved_model_name and ckpt > checkpoint:
                checkpoint = ckpt
    path = "{}/{}@{}.ckpt".format(__model_path__, saved_model_name, checkpoint)
    if not os.path.exists(path):
        log("Checkpoint not found.")
    else:
        log("Checkpoint found, restoring from {}".format(checkpoint))
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        else:
            model.load_state_dict(torch.load(path))
    return checkpoint


def save_model(model, saved_model_name, checkpoint):
    if not os.path.exists(__model_path__):
        os.makedirs(__model_path__, exist_ok=True)
    if checkpoint == -1:
        checkpoint = 0
    torch.save(model.state_dict(), "{}/{}@{}.ckpt".format(
        __model_path__, saved_model_name, checkpoint))
    return checkpoint + 1


class TimingSaver:
    def __init__(self, model, model_name, seconds, init_ckpt=-1):
        self.model = model
        self.model_name = model_name
        self.seconds = seconds
        self.last_time = time.time()
        self.ckpt = load_model(model=self.model,
                               saved_model_name=self.model_name,
                               checkpoint=init_ckpt)

    def save(self):
        curr_time = time.time()
        if curr_time - self.last_time > self.seconds:
            self.ckpt = save_model(model=self.model,
                                   saved_model_name=self.model_name,
                                   checkpoint=self.ckpt)
            self.last_time = curr_time


def ten2var(x):
    return gpu(torch.autograd.Variable(x))


def long2var(x):
    return gpu(torch.autograd.Variable(torch.LongTensor(x)))


def float2var(x):
    return gpu(torch.autograd.Variable(torch.FloatTensor(x)))


def var2list(x):
    return x.cpu().data.numpy().tolist()


def var2num(x):
    return x.cpu().data[0]


def load_word2vec(embedding: torch.nn.Embedding,
                  word_dict: Dict[str, int],
                  word2vec_path,
                  norm=True,
                  cached_name=None):
    def __norm2one(vec):
        root_sum_square = np.sqrt(np.sum(np.square(vec)))
        return vec / root_sum_square

    cache = "{}{}".format(cached_name, ".norm" if norm else "")
    if cached_name and exist_var(cache):
        log("Load vocab from cache {}".format(cache))
        pre_embedding = load_var(cache)
    else:
        log("Load vocab from {}".format(word2vec_path))
        scale = np.sqrt(3.0 / embedding.embedding_dim)
        pre_embedding = np.random.normal(0, 1, embedding.weight.size())
        wordvec_file = open(word2vec_path, errors='ignore')
        # x = 0
        found = 0
        for line in wordvec_file.readlines():
            # x += 1
            # log("Process line {} in file {}".format(x, word2vec_path))
            split = re.split(r"\s+", line.strip())
            # for word2vec, the first line is meta info: (NUMBER, SIZE)
            if len(split) < 10:
                continue
            word = split[0]
            if word in word_dict:
                found += 1
                emb = list(map(float, split[1:]))
                if norm:
                    pre_embedding[word_dict[word]] = __norm2one(np.array(emb))
                else:
                    pre_embedding[word_dict[word]] = np.array(emb)
        log("Pre_train match case: {:.4f}".format(found / len(word_dict)))
        if cached_name:
            save_var(pre_embedding, cache)
    embedding.weight.data.copy_(torch.from_numpy(pre_embedding))


def flip_by_length(inputs, lengths):
    rev_inputs = []
    for it_id, it_input in enumerate(inputs):
        it_len = lengths[it_id]
        rev_input = torch.cat([
            it_input.index_select(0, torch.tensor(list(reversed(range(it_len)))).to(inputs.device)),
            torch.zeros_like(it_input[it_len:]).to(inputs.device)
        ])
        rev_inputs.append(rev_input)
    rev_inputs = torch.stack(rev_inputs)
    return rev_inputs


def focal_loss(inputs, targets, gamma=2, alpha=None, size_average=True):
    N = inputs.size(0)
    C = inputs.size(1)
    P = F.softmax(inputs, dim=1)

    class_mask = inputs.data.new(N, C).fill_(0)
    ids = targets.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)
    if alpha is None:
        alpha = torch.ones(C, 1).to(inputs.device)
    alpha = alpha[ids.data.view(-1)]

    probs = (P * class_mask).sum(1).view(-1, 1)

    log_p = probs.log()

    batch_loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p

    if size_average:
        loss = batch_loss.mean()
    else:
        loss = batch_loss.sum()
    return loss
