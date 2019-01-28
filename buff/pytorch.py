from .public import *
import torch
from torch.nn.utils.rnn import PackedSequence
import torch.nn.functional as F
from .logging import log
import pandas

__model_path__ = "saved/models"


def cast_list(array):
    if isinstance(array, torch.Tensor):
        return cast_list(array.detach().cpu().numpy())
    if isinstance(array, list):
        return cast_list(np.array(array))
    if isinstance(array, np.ndarray):
        return array.squeeze().tolist()


def allocate_cuda_device(cuda_idx) -> torch.device:
    if torch.cuda.is_available() and cuda_idx >= 0:
        device = torch.device("cuda:{}".format(cuda_idx))
    else:
        device = torch.device("cpu")
    return device


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


class ModelManager:
    def __init__(self, model, model_name, seconds=-1, init_ckpt=-1):
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
    cache = "{}{}".format(cached_name, ".norm" if norm else "")
    if cached_name and exist_var(cache):
        log("Load vocab from cache {}".format(cache))
        pre_embedding = load_var(cache)
    else:
        log("Load vocab from {}".format(word2vec_path))
        pre_embedding = np.random.normal(0, 1, embedding.weight.size())
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
    embedding.weight.data.copy_(torch.from_numpy(pre_embedding))


def load_word_and_its_vec(word2vec_path,
                          norm=True,
                          cached_name=None):
    cache = "{}{}".format(cached_name, ".norm" if norm else "")
    if cached_name and exist_var(cache):
        log("Load vocab from cache {}".format(cache))
        word2idx, idx2word, embeddings = load_var(cache)
    else:
        log("Load vocab from {}".format(word2vec_path))
        csv = pandas.read_csv(word2vec_path, sep="\\s+")
        words = csv.values[:, 0]
        embeddings = csv.values[:, 1:]
        word2idx = {}
        for idx, word in enumerate(words):
            word2idx[word] = idx
        idx2word = {v: k for k, v in word2idx.items()}
        print("Embedding num: {} dim: {} mean: {:.3f} std: {:.3f}".format(
            *embeddings.shape, embeddings.mean(), embeddings.std()
        ))
        if norm:
            embeddings = embeddings / embeddings.std()
        if cached_name:
            save_var((word2idx, idx2word, embeddings), cache)
    return word2idx, idx2word, embeddings


def show_mean_std(tensor):
    print("Mean {:.4f} Std {:.4f}".format(tensor.mean().item(),
                                          tensor.std().item()))


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


def focal_loss(inputs,
               targets,
               gamma=2, alpha=None, reduction="mean"):
    batch_size = inputs.size(0)
    num_classes = inputs.size(1)
    prob = F.softmax(inputs, dim=1).clamp(1e-10, 1.)
    # prob = inputs.exp()

    class_mask = inputs.data.new(batch_size, num_classes).fill_(0)
    ids = targets.view(-1, 1)
    class_mask.scatter_(1, ids.data, 1.)
    if alpha is None:
        alpha = torch.ones(num_classes, 1).to(inputs.device)
    alpha = alpha[ids.data.view(-1)]

    probs = (prob * class_mask).sum(1).view(-1, 1)

    log_p = probs.log()

    batch_loss = -alpha * (torch.pow((1 - probs), gamma)) * log_p

    if reduction == "mean":
        loss = batch_loss.mean()
    elif reduction == "sum":
        loss = batch_loss.sum()
    elif reduction == "zheng":
        pred = torch.argmax(inputs, dim=1)
        ce_mask = pred != targets
        loss = torch.mean(batch_loss * ce_mask)
    elif reduction == "none":
        loss = batch_loss
    else:
        raise Exception()
    return loss


class NonLinearLayerWithRes(torch.nn.Module):
    def __init__(self, d_in, d_hidden, dropout):
        super(NonLinearLayerWithRes, self).__init__()
        self.fc1 = torch.nn.Linear(d_in, d_hidden)
        self.fc2 = torch.nn.Linear(d_hidden, d_in)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x):
        out = self.fc2(F.relu(self.fc1(x)))
        out += x
        out = self.drop(out)
        # out = torch.nn.LayerNorm(out)
        return out
