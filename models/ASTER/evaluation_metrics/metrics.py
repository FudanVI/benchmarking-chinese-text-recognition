from __future__ import absolute_import

import numpy as np
import editdistance
import string
import math

import torch
import torch.nn.functional as F

from utils import to_torch, to_numpy
from utils.radical_dict import *
import zhconv

def _normalize_text(text):
  # text = ''.join(filter(lambda x: x in (string.digits + string.ascii_letters), text))
  # return text.lower()
  path = "./data/benchmark_new.txt"
  with open(path, "r") as f:
    voc = list(f.readlines()[0])
  text = ''.join(filter(lambda x: x in (voc), text))
  return text.lower()

def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
      inside_code = ord(uchar)
      if inside_code == 12288:  # 全角空格直接转换
        inside_code = 32
      elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
        inside_code -= 65248
      rstring += chr(inside_code)
    return rstring

def get_str_list(output, target, dataset=None):
  output=output.argmax(dim=2)
  end_label = alp2num_character['$']
  unknown_label = alp2num_character['PADDING']
  num_samples, max_len_labels = output.size()

  assert num_samples == target.size(0) and max_len_labels == target.size(1)
  output = to_numpy(output)
  target = to_numpy(target)

  # list of char list
  pred_list, targ_list = [], []
  for i in range(num_samples):
    pred_list_i = []
    for j in range(max_len_labels):
      if output[i, j] != end_label:
        if output[i, j] != unknown_label:
          pred_list_i.append(num2alp_character[output[i, j]])
      else:
        break
    pred_list.append(pred_list_i)

  for i in range(num_samples):
    targ_list_i = []
    for j in range(max_len_labels):
      if target[i, j] != end_label:
        if target[i, j] != unknown_label:
          targ_list_i.append(num2alp_character[target[i, j]])
      else:
        break
    targ_list.append(targ_list_i)

  # 繁体转简体
  pred_list = [zhconv.convert(strQ2B(pred),'zh-cn') for pred in pred_list]
  targ_list = [zhconv.convert(strQ2B(targ),'zh-cn') for targ in targ_list]

  pred_list = [_normalize_text(pred) for pred in pred_list]
  targ_list = [_normalize_text(targ) for targ in targ_list]

  return pred_list, targ_list



