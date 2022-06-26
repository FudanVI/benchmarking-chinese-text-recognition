from __future__ import absolute_import

# import sys
# sys.path.append('./')

import os
from PIL import Image, ImageFile
import numpy as np
import random
import json
import lmdb
import sys
import six

import torch
from torch.utils import data
from torch.utils.data import sampler
from torchvision import transforms

from utils.labelmaps import get_vocabulary, labels2strs
from utils import to_numpy

ImageFile.LOAD_TRUNCATED_IMAGES = True
import cv2

from config import get_args
global_args = get_args(sys.argv[1:])

if global_args.run_on_remote:
  import moxing as mox


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
      inside_code = ord(uchar)
      if inside_code == 12288:
        inside_code = 32
      elif (inside_code >= 65281 and inside_code <= 65374):
        inside_code -= 65248

      rstring += chr(inside_code)
    return rstring


from utils.radical_dict import *

class LmdbDataset(data.Dataset):
  def __init__(self, root, voc_type, max_len, num_samples,voc_path="", transform=None):
    super(LmdbDataset, self).__init__()
    self.voc_path=voc_path
    if global_args.run_on_remote:
      dataset_name = os.path.basename(root)
      data_cache_url = "/cache/%s" % dataset_name
      if not os.path.exists(data_cache_url):
        os.makedirs(data_cache_url)
      if mox.file.exists(root):
        mox.file.copy_parallel(root, data_cache_url)
      else:
        raise ValueError("%s not exists!" % root)
      
      self.env = lmdb.open(data_cache_url, max_readers=32, readonly=True)
    else:
      self.env = lmdb.open(root, max_readers=32, readonly=True)

    assert self.env is not None, "cannot create lmdb from %s" % root
    self.txn = self.env.begin()

    self.voc_type = voc_type
    self.transform = transform
    self.max_len = max_len
    self.nSamples = int(self.txn.get(b"num-samples"))
    self.nSamples = min(self.nSamples, num_samples)

    # assert voc_type in ['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS']
    # self.EOS = 'EOS'
    # self.PADDING = 'PADDING'
    # self.UNKNOWN = 'UNKNOWN'
    # self.voc = get_vocabulary(voc_type, EOS=self.EOS, PADDING=self.PADDING, UNKNOWN=self.UNKNOWN)
    # self.char2id = dict(zip(self.voc, range(len(self.voc))))
    # self.id2char = dict(zip(range(len(self.voc)), self.voc))

    self.rec_num_classes = len(alp2num_character)
    self.lowercase = (voc_type == 'LOWERCASE')

  def __len__(self):
    return self.nSamples

  def __getitem__(self, index):
    assert index <= len(self), 'index range error'
    index += 1
    img_key = b'image-%09d' % index
    imgbuf = self.txn.get(img_key)

    buf = six.BytesIO()
    buf.write(imgbuf)
    buf.seek(0)
    try:
      img = Image.open(buf).convert('RGB')
      # img = Image.open(buf).convert('L')
      # img = img.convert('RGB')
    except IOError:
      print('Corrupted image for %d' % index)
      return self[index + 1]

    # reconition labels
    label_key = b'label-%09d' % index
    word = self.txn.get(label_key).decode()
    if self.lowercase:
      word = word.lower()

    label = np.full((self.max_len,), alp2num_character['PADDING'], dtype=np.int)
    label_list = []
    for char in word:
      if char in alp2num_character.keys():
        label_list.append(alp2num_character[char])
      else:
        ## add the unknown token
        # print('{0} is out of vocabulary.'.format(char))
        label_list.append(alp2num_character['UNKNOW'])
    ## add a stop token
    label_list = label_list + [alp2num_character['END']]
    label[:len(label_list)] = np.array(label_list)

    if len(label) <= 0:
      return self[index + 1]

    # label length
    label_len = len(label_list)


    ## fill with the padding token

    ## add a stop token
    word = word + '$'
    word = strQ2B(word)
    if self.transform is not None:
      img = self.transform(img)
    return img, word,label_len,label


class ResizeNormalize(object):
  def __init__(self, size, interpolation=Image.BILINEAR):
    self.size = size
    self.interpolation = interpolation
    self.toTensor = transforms.ToTensor()

  def __call__(self, img):
    img = img.resize(self.size, self.interpolation)
    img = self.toTensor(img)
    img.sub_(0.5).div_(0.5)
    return img


class RandomSequentialSampler(sampler.Sampler):

  def __init__(self, data_source, batch_size):
    self.num_samples = len(data_source)
    self.batch_size = batch_size

  def __len__(self):
    return self.num_samples

  def __iter__(self):
    n_batch = len(self) // self.batch_size
    tail = len(self) % self.batch_size
    index = torch.LongTensor(len(self)).fill_(0)
    for i in range(n_batch):
      random_start = random.randint(0, len(self) - self.batch_size)
      batch_index = random_start + torch.arange(0, self.batch_size)
      index[i * self.batch_size:(i + 1) * self.batch_size] = batch_index
    # deal with tail
    if tail:
      random_start = random.randint(0, len(self) - self.batch_size)
      tail_index = random_start + torch.arange(0, tail)
      index[(i + 1) * self.batch_size:] = tail_index

    return iter(index.tolist())


class AlignCollate(object):

  def __init__(self, imgH=32, imgW=100, keep_ratio=False, min_ratio=1):
    self.imgH = imgH
    self.imgW = imgW
    self.keep_ratio = keep_ratio
    self.min_ratio = min_ratio

  def __call__(self, batch):
    images, labels, lengths,labels_ = zip(*batch)
    b_lengths = torch.IntTensor(lengths)
    b_labels = torch.IntTensor(labels_)

    imgH = self.imgH
    imgW = self.imgW
    if self.keep_ratio:
      ratios = []
      for image in images:
        w, h = image.size
        ratios.append(w / float(h))
      ratios.sort()
      max_ratio = ratios[-1]
      imgW = int(np.floor(max_ratio * imgH))
      imgW = max(imgH * self.min_ratio, imgW)  # assure imgH >= imgW
      imgW = min(imgW, 400)

    transform = ResizeNormalize((imgW, imgH))
    images = [transform(image) for image in images]
    b_images = torch.stack(images)

    return b_images, labels, b_lengths,b_labels

if __name__ == "__main__":
    debug()