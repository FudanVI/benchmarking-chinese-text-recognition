from __future__ import absolute_import

import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

def to_contiguous(tensor):
  if tensor.is_contiguous():
    return tensor
  else:
    return tensor.contiguous()

def _assert_no_grad(variable):
  assert not variable.requires_grad, \
    "nn criterions don't compute the gradient w.r.t. targets - please " \
    "mark these variables as not requiring gradients"

class EmbeddingRegressionLoss(nn.Module):
    def __init__(self,
                 weight=None,
                 size_average=True,
                 ignore_index=-100,
                 sequence_normalize=False,
                 sample_normalize=True,
                 loss_func='cosin'):
        super(EmbeddingRegressionLoss, self).__init__()
        self.weight = weight
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.sequence_normalize = sequence_normalize
        self.sample_normalize = sample_normalize
        # self.loss_func = torch.nn.MSELoss()
        self.is_cosin_loss = False
        if loss_func == 'smooth_l1':
            self.loss_func = torch.nn.SmoothL1Loss()
        elif loss_func == 'cosin':
            self.loss_func = torch.nn.CosineEmbeddingLoss()
            self.is_cosin_loss = True

    def forward(self, input, target):
        _assert_no_grad(target)

        if not self.is_cosin_loss:
            Loss = self.loss_func(input, target)
        else:
            label_target = torch.ones(input.size(0)).cuda()
            Loss = self.loss_func(input, target, label_target)

        return Loss
    def logistic_dot_loss(self, input, target):
        dot_result = torch.mm(input, target.t())
        _diagaonal = dot_result.diagonal()
        logistic_loss = torch.log(1 + torch.exp(-1 * _diagaonal))

        # logistic_loss = torch.mean(logistic_loss, dim=0)

        return logistic_loss
        # _trace = torch.trace(dot_result)
        # loss = _trace / input.size(0)
        #
        # logistic_loss = nn.sigmoid(loss)

