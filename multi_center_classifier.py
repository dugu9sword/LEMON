import torch
import numpy as np
import torch.nn.functional as F
import math


def aggregate_class_mask(num_classes, k_center):
    w = np.zeros(shape=(num_classes * k_center, num_classes))
    for i in range(num_classes * k_center):
        w[i][i // k_center] = 1.
    return w


def in_class_mask(num_classes, k_center):
    w = np.zeros(shape=(num_classes * k_center, num_classes * k_center))
    for i in range(num_classes * k_center):
        i_cls = i // k_center
        w[i][i_cls * k_center: i_cls * k_center + k_center] = 1.
    return w


def between_class_mask(num_classes, k_center):
    return 1. - in_class_mask(num_classes, k_center)


def reset_parameters(weight, bias):
    torch.nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    if bias is not None:
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(weight)
        bound = 1 / math.sqrt(fan_in)
        torch.nn.init.uniform_(bias, -bound, bound)


class KCenterClassifier(torch.nn.Module):

    def __init__(self, num_classes, k_center, dim, aggregate=False):
        super(KCenterClassifier, self).__init__()
        self.num_classes = num_classes
        self.k_center = k_center
        self._width = num_classes * k_center

        self.mask = torch.nn.Parameter(torch.tensor(aggregate_class_mask(num_classes, k_center)).float(),
                                       requires_grad=False)
        self.in_class_mask = torch.nn.Parameter(torch.tensor(in_class_mask(num_classes, k_center)).float(),
                                                requires_grad=False)
        self.between_class_mask = torch.nn.Parameter(torch.tensor(between_class_mask(num_classes, k_center)).float(),
                                                     requires_grad=False)
        self.identity = torch.nn.Parameter(torch.eye(self.num_classes * self.k_center).float(),
                                           requires_grad=False)

        self.label_weight = torch.nn.Parameter(torch.Tensor(num_classes * k_center, dim))
        self.label_bias = torch.nn.Parameter(torch.Tensor(num_classes * k_center))

        reset_parameters(weight=self.label_weight, bias=self.label_bias)

        # std = 1. / math.sqrt(self.label_weight.size(0))
        # self.label_weight.data.uniform_(-std, std)
        # self.label_bias.data.uniform_(-std, std)

    def forward(self, x,
                return_logits=True):
        logits = (x @ self.label_weight.t() + self.label_bias)
        logits = logits @ self.mask
        if return_logits:
            return logits
        else:
            return F.softmax(logits, dim=1)

    def aux_loss(self):
        delta = self.label_weight @ self.label_weight.t() - self.identity
        in_cls_loss = torch.sum(delta ** 2 * self.in_class_mask)
        between_cls_loss = torch.sum(delta ** 2 * self.between_class_mask)
        return in_cls_loss, between_cls_loss


# dynamic_soft = KCenterClassifier(5, 2, 1024)
# print(dynamic_soft.aux_loss())


class WTFClassifier(torch.nn.Module):
    def __init__(self, num_classes, k_center, dim):
        super(WTFClassifier, self).__init__()
        self.label_weight = torch.nn.Parameter(torch.Tensor(num_classes * k_center, dim))
        self.label_bias = torch.nn.Parameter(torch.Tensor(num_classes * k_center))
        self._label_weight_3d_view = self.label_weight.view(num_classes, k_center, dim)
        self.k_center = k_center
        self.num_classes = num_classes
        self._dim = dim

        self.num_ks = [1 for _ in range(num_classes)]

        reset_parameters(self.label_weight, self.label_bias)
        # std = 1. / math.sqrt(self.label_weight.size(0))
        # self.label_weight.data.uniform_(-std, std)
        # self.label_bias.data.uniform_(-std, std)

    def add(self, cls_idx, x):
        self._label_weight_3d_view.data[cls_idx, self.num_ks[cls_idx]] = x.data[:]
        self.num_ks[cls_idx] = self.num_ks[cls_idx] + 1

    def is_full(self, class_idx):
        return self.num_ks[class_idx] == self.k_center

    def extract_max_score(self, score):
        score, _ = torch.max(score.view(-1,
                                        self.num_classes,
                                        self.k_center), dim=2)
        return score

    def forward(self, x, return_logits=True):
        # x: batch_size * dim
        batch_size = x.size(0)
        att_score = x @ self.label_weight.t() + self.label_bias
        mask = torch.zeros(batch_size, self.num_classes, self.k_center,
                           dtype=torch.uint8,
                           device=self.label_weight.device)
        for cls_idx, k in enumerate(self.num_ks):
            mask[:, cls_idx, k:] = 1
        mask = mask.view(batch_size, -1)
        att_score.masked_fill_(mask, -np.inf)
        if return_logits:
            return att_score
        else:
            return F.softmax(att_score, dim=1)

    def compute_loss(self, nk_logits, targets, m=1):
        batch_size = nk_logits.size(0)
        assert self.num_classes * self.k_center == nk_logits.size(1)
        batch_loss = torch.zeros_like(targets, dtype=torch.float)

        max_ctr = nk_logits.argmax(dim=1, keepdim=True)

        # print(nk_logits[0].view(-1, 5))
        # preds = max_ctr.view(-1).numpy().tolist()
        # print(set(sorted(preds)))

        for i in range(batch_size):
            it_input = nk_logits[i]
            it_target = targets[i]
            it_pred_ctr = max_ctr[i][0]
            it_pred_cls = it_pred_ctr // self.k_center

            if it_pred_cls == targets[i]:
                pass
            else:
                gold_cls_max_ctr_score = it_input[it_target * self.k_center
                                                  : (it_target + 1) * self.k_center].max()
                batch_loss[i] = m - gold_cls_max_ctr_score + it_input[it_pred_ctr]
                # gold_cls_random_ctr_score = it_input[it_target * k: (it_target + 1) * k][np.random.randint(k)]
                # batch_loss += m - gold_cls_random_ctr_score + it_input[it_pred_ctr]
                # gold_cls_avg_ctr_score = it_input[it_target * k: (it_target + 1) * k].mean()
                # batch_loss += m - gold_cls_avg_ctr_score + it_input[it_pred_ctr]

        return batch_loss

    def add_max_loss_feature(self,
                             feature,
                             target,
                             batch_loss):
        max_loss_idx = torch.argmax(batch_loss)
        max_loss_cls = target[max_loss_idx]
        if not self.is_full(max_loss_cls):
            print('*** Add a new feature to cls', max_loss_cls.item())
            self.add(max_loss_cls, feature[max_loss_idx])
        else:
            print('*** Full cls ', max_loss_cls.item())

# dynamic_soft = DynamicCenterClassifier(5, 3, 100)
# dynamic_soft.add(0, torch.randn(100))
# print(dynamic_soft.is_full(0))
#
# dynamic_soft.add(0, torch.randn(100))
# print(dynamic_soft.is_full(0))
# # dynamic_soft.add(1, torch.randn(100))
# # dynamic_soft.add(2, torch.randn(100))
# prob = dynamic_soft(torch.randn(3, 100))
# print(prob.size())
# print(prob)
