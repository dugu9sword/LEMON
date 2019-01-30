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

    def __init__(self, num_classes, k_center, dim):
        super(KCenterClassifier, self).__init__()
        self._num_classes = num_classes
        self._k = k_center
        self._width = num_classes * k_center

        self.mask = torch.nn.Parameter(torch.tensor(aggregate_class_mask(num_classes, k_center)).float(),
                                       requires_grad=False)
        self.in_class_mask = torch.nn.Parameter(torch.tensor(in_class_mask(num_classes, k_center)).float(),
                                                requires_grad=False)
        self.between_class_mask = torch.nn.Parameter(torch.tensor(between_class_mask(num_classes, k_center)).float(),
                                                     requires_grad=False)
        self.identity = torch.nn.Parameter(torch.eye(self._num_classes * self._k).float(),
                                           requires_grad=False)

        self.label_weight = torch.nn.Parameter(torch.Tensor(num_classes * k_center, dim))
        self.label_bias = torch.nn.Parameter(torch.Tensor(num_classes * k_center))

        reset_parameters(weight=self.label_weight, bias=self.label_bias)

        # std = 1. / math.sqrt(self.label_weight.size(0))
        # self.label_weight.data.uniform_(-std, std)
        # self.label_bias.data.uniform_(-std, std)

    def forward(self, x, return_logits=True):
        logits = (x @ self.label_weight.t() + self.label_bias) @ self.mask
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


class DynamicCenterClassifier(torch.nn.Module):
    def __init__(self, num_classes, max_k, dim):
        super(DynamicCenterClassifier, self).__init__()
        self.label_weight = torch.nn.Parameter(torch.Tensor(num_classes * max_k, dim))
        self.label_bias = torch.nn.Parameter(torch.Tensor(num_classes * max_k))
        self._label_weight_3d_view = self.label_weight.view(num_classes, max_k, dim)
        self._max_k = max_k
        self._num_classes = num_classes
        self._dim = dim

        self.num_ks = [1 for _ in range(num_classes)]

        std = 1. / math.sqrt(self.label_weight.size(0))
        self.label_weight.data.uniform_(-std, std)
        self.label_bias.data.uniform_(-std, std)

    def add(self, cls_idx, x):
        self._label_weight_3d_view.data[cls_idx, self.num_ks[cls_idx]] = x.data[:]
        self.num_ks[cls_idx] = self.num_ks[cls_idx] + 1

    def is_full(self, class_idx):
        return self.num_ks[class_idx] == self._max_k

    def forward(self, x, return_logits=True):
        # x: batch_size * dim
        batch_size = x.size(0)
        att_score = x @ self.label_weight.t() + self.label_bias
        mask = torch.zeros(batch_size, self._num_classes, self._max_k,
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
