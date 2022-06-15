from __future__ import print_function, absolute_import
import time
from time import gmtime, strftime
from datetime import datetime
from collections import OrderedDict

import torch

import numpy as np
from random import randint
from PIL import Image
import sys

import evaluation_metrics
from evaluation_metrics.metrics import get_str_list
from utils.meters import AverageMeter
from torchvision import transforms

metrics_factory = evaluation_metrics.factory()
import cv2

toImage = transforms.ToPILImage()
toTensor = transforms.ToTensor()

from config import get_args
from utils.radical_dict import converter
global_args = get_args(sys.argv[1:])


class BaseEvaluator(object):
    def __init__(self, model, metric, use_cuda=True):
        super(BaseEvaluator, self).__init__()
        self.model = model
        self.metric = metric
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def evaluate(self, data_loader, step=1, print_freq=1, tfLogger=None, dataset=None, vis_dir=None):
        self.model.eval()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        # forward the network
        images, outputs, targets, losses = [], {}, [], []
        file_names = []

        end = time.time()
        total_num = 0
        total_true = 0
        index = 0
        for i, inputs in enumerate(data_loader):
            outputs = {}
            outputs['pred_rec'] = {}

            data_time.update(time.time() - end)

            input_dict = self._parse_data(inputs)
            rec_pred, att_map, _ ,_ = self._forward(input_dict)
            outputs['pred_rec'] = rec_pred

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('[{}]\t'
                          'Evaluation: [{}/{}]\t'
                          'Time {:.3f} ({:.3f})\t'
                          'Data {:.3f} ({:.3f})\t'
                          # .format(strftime("%Y-%m-%d %H:%M:%S", gmtime()),
                          .format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                  i + 1, len(data_loader),
                                  batch_time.val, batch_time.avg,
                                  data_time.val, data_time.avg))

            pred_list, targ_list = get_str_list(rec_pred, input_dict['rec_targets'], None)
            acc_list = [(pred == targ) for pred, targ in zip(pred_list, targ_list)]
            total_true += sum(acc_list)
            total_num += len(acc_list)

        accuracy = 1.0 * total_true / total_num
        print("test accury:{}".format(accuracy))
        return accuracy

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs):
        raise NotImplementedError


class Evaluator(BaseEvaluator):
  def _parse_data(self, inputs):
    input_dict = {}
    if global_args.evaluate_with_lexicon:
      imgs, labels, lengths, file_name,embeds = inputs
    else:
      imgs, labels, lengths,labels_,embeds = inputs

    with torch.no_grad():
      images = imgs.to(self.device)

    input_dict['images'] = images
    length, text_input, text_all, length_radical, radical_input, radical_all, string_label = converter(labels)
    if text_input is not None:
      text_input = text_input.to(self.device)

    input_dict['rec_targets'] = text_input.cuda()
    input_dict['rec_lengths'] = length.cuda()
    input_dict['radical_input'] = radical_input
    input_dict['rec_embeds'] = embeds.cuda()

    input_dict['length_radical'] = length_radical
    input_dict['radical_gt'] = radical_all
    if global_args.evaluate_with_lexicon:
      input_dict['file_name'] = file_name
    return input_dict

  def _forward(self, input_dict):
    self.model.eval()
    return self.model(input_dict)