from __future__ import print_function, absolute_import
import time
from time import gmtime, strftime
from datetime import datetime
import gc
import os.path as osp
import sys
from PIL import Image
import numpy as np

import torch
from torchvision import transforms

from utils import to_numpy
from utils.meters import AverageMeter
from utils.serialization import load_checkpoint, save_checkpoint


from utils.radical_dict import *
from loss.sequenceCrossEntropyLoss import SequenceCrossEntropyLoss
from config import get_args
global_args = get_args(sys.argv[1:])
from utils.radical_dict import converter
from loss.embeddingRegressionLoss import EmbeddingRegressionLoss

class BaseTrainer(object):
  def __init__(self, model, metric, logs_dir, iters=0, best_res=-1, grad_clip=-1, use_cuda=True, loss_weights={}):
    super(BaseTrainer, self).__init__()
    self.model = model
    self.metric = metric
    self.logs_dir = logs_dir
    self.iters = iters
    self.best_res = best_res
    self.grad_clip = grad_clip
    self.use_cuda = use_cuda
    self.loss_weights = loss_weights

    self.device = torch.device("cuda" if use_cuda else "cpu")
    self.rec_crit = SequenceCrossEntropyLoss()
    self.embed_crit = EmbeddingRegressionLoss(loss_func='cosin')
    self.radical_crit = torch.nn.CrossEntropyLoss()

  def train(self, epoch, data_loader, optimizer, current_lr=0.0, 
            print_freq=100, train_tfLogger=None, is_debug=False,
            evaluator=None, test_loader=None, eval_tfLogger=None,
            test_dataset=None, test_freq=1000):

    self.model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    test_freq = len(data_loader) * 1
    #训练前测试
    # evaluator.evaluate(test_loader, step=self.iters, tfLogger=eval_tfLogger, dataset=test_dataset)
    for i, inputs in enumerate(data_loader):
      # try:
        return_dict = {}
        return_dict['losses'] = {}
        return_dict['output'] = {}

        self.model.train()
        self.iters += 1

        data_time.update(time.time() - end)

        input_dict = self._parse_data(inputs)
        rec_pred, att_map, radical_decoder_result,embedding_vectors = self._forward(input_dict)

        # 字符分支损失
        loss_rec = self.rec_crit(rec_pred, input_dict['rec_targets'], input_dict['rec_lengths'])
        return_dict['losses']['loss_rec'] = loss_rec

        # embeddin loss
        loss_embed = self.embed_crit(embedding_vectors, input_dict['rec_embeds'])
        return_dict['losses']['loss_embed'] = loss_embed

        # 部首分支损失
        length_radical = input_dict['length_radical']
        total_radical_length = torch.sum(length_radical.view(-1)).data
        probs_res_radical = torch.zeros((total_radical_length, len(alphabet_radical_raw))).type_as(
          radical_decoder_result.data)
        start = 0
        for index, length in enumerate(length_radical.view(-1)):
          # print("index, length", index,length)
          length = length.data
          if length == 0:
            continue

          probs_res_radical[start:start + length, :] = radical_decoder_result[index, 0:0 + length, :]
          start = start + length

        return_dict["radical_pred"] = probs_res_radical

        return_dict['losses']["radical_loss"] = self.radical_crit(return_dict["radical_pred"], input_dict['radical_gt'])

        # pytorch0.4 bug on gathering scalar(0-dim) tensors
        for k, v in return_dict['losses'].items():
          return_dict['losses'][k] = v.unsqueeze(0)

        batch_size = input_dict['images'].size(0)

        total_loss = 0
        loss_dict = {}
        for k, loss in return_dict['losses'].items():
          loss = loss.mean(dim=0, keepdim=True)
          total_loss += self.loss_weights[k] * loss
          loss_dict[k] = loss.item()

        losses.update(total_loss.item(), batch_size)

        optimizer.zero_grad()
        total_loss.backward()
        if self.grad_clip > 0:
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        optimizer.step()


        batch_time.update(time.time() - end)
        end = time.time()

        if self.iters % 100 == 0:
          print('[{}]\t'
                'Epoch: [{}][{}/{}]\t'
                'Time {:.3f} ({:.3f})\t'
                'Data {:.3f} ({:.3f})\t'
                'Loss {:.3f} ({:.3f})\t'
                'Embedding_loss {}\t'
                'Radical_loss {}\t'
                # .format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        epoch, i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg,
                        losses.val, losses.avg,
                        return_dict['losses']['loss_embed'],return_dict['losses']["radical_loss"]))

        # ====== TensorBoard logging ======#
        if self.iters % print_freq * 10 == 0:
          if train_tfLogger is not None:
            step = epoch * len(data_loader) + (i + 1)
            info = {
              'lr': current_lr,
              'loss': total_loss.item(),  # this is total loss
            }
            ## add each loss
            for k, loss in loss_dict.items():
              info[k] = loss
            for tag, value in info.items():
              train_tfLogger.scalar_summary(tag, value, step)


        # ====== evaluation ======#
        if self.iters % test_freq == 0:
          if 'loss_rec' not in return_dict['losses']:
            is_best = True
            # self.best_res is alwarys equal to 1.0
            self.best_res = evaluator.evaluate(test_loader, step=self.iters, tfLogger=eval_tfLogger,
                                               dataset=test_dataset)
          else:
            res = evaluator.evaluate(test_loader, step=self.iters, tfLogger=eval_tfLogger, dataset=test_dataset)

            if self.metric == 'accuracy':
              is_best = res > self.best_res
              self.best_res = max(res, self.best_res)
            elif self.metric == 'editdistance':
              is_best = res < self.best_res
              self.best_res = min(res, self.best_res)
            else:
              raise ValueError("Unsupported evaluation metric:", self.metric)

            print('\n * Finished iters {:3d}  accuracy: {:5.1%}  best: {:5.1%}{}\n'.
                  format(self.iters, res, self.best_res, ' *' if is_best else ''))

          save_checkpoint({
            'state_dict': self.model.module.state_dict(),
            'iters': self.iters,
            'best_res': self.best_res,
          }, is_best, fpath=osp.join(self.logs_dir, 'checkpoint.pth.tar'))


  def _parse_data(self, inputs):
    raise NotImplementedError

  def _forward(self, inputs, targets):
    raise NotImplementedError


class Trainer(BaseTrainer):
  def _parse_data(self, inputs):
    input_dict = {}
    imgs, labels, lengths,label_,embeds = inputs
    images = imgs.to(self.device)


    input_dict['images'] = images
    length, text_input, text_all, length_radical, radical_input, radical_all, string_label = converter(labels)

    if radical_input is not None:
      radical_input=radical_input.to(self.device)
    input_dict['rec_targets'] = text_input.cuda()
    input_dict['rec_lengths'] = length.cuda()
    input_dict['radical_input'] = radical_input
    input_dict['rec_embeds'] = embeds.cuda()

    input_dict['length_radical'] = length_radical
    input_dict['radical_gt'] = radical_all
    return input_dict

  def _forward(self, input_dict):
    self.model.train()
    return self.model(input_dict)