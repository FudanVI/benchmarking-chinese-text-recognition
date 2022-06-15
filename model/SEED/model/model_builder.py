from __future__ import absolute_import

from PIL import Image
import numpy as np
from collections import OrderedDict
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init

from . import create
from .attention_recognition_head import AttentionRecognitionHead
from .tps_spatial_transformer import TPSSpatialTransformer
from .stn_head import STNHead
from .radical_decoder import RadicalBranch
from .embedding_head import Embedding

from config import get_args
global_args = get_args(sys.argv[1:])

from utils.radical_dict import *

class ModelBuilder(nn.Module):
  """
  This is the integrated model.
  """
  def __init__(self, arch, rec_num_classes, sDim, attDim, max_len_labels, eos,time_step=32, STN_ON=False):
    super(ModelBuilder, self).__init__()

    self.arch = arch
    self.rec_num_classes = rec_num_classes
    self.sDim = sDim
    self.attDim = attDim
    self.max_len_labels = max_len_labels
    self.eos = eos
    self.STN_ON = STN_ON
    self.time_step = time_step
    self.tps_inputsize = global_args.tps_inputsize

    self.encoder = create(self.arch,
                      with_lstm=global_args.with_lstm,
                      n_group=global_args.n_group)
    encoder_out_planes = self.encoder.out_planes


    self.radical_decoder = RadicalBranch(radical_alphabet_len=len(alphabet_radical_raw))

    self.decoder = AttentionRecognitionHead(
                      num_classes=rec_num_classes,
                      in_planes=encoder_out_planes,
                      sDim=sDim,
                      attDim=attDim,
                      max_len_labels=max_len_labels)

    self.embeder = Embedding(self.time_step, encoder_out_planes)

    if self.STN_ON:
      self.tps = TPSSpatialTransformer(
        output_image_size=tuple(global_args.tps_outputsize),
        num_control_points=global_args.num_control_points,
        margins=tuple(global_args.tps_margins))
      self.stn_head = STNHead(
        in_planes=3,
        num_ctrlpoints=global_args.num_control_points,
        activation=global_args.stn_activation)

  def forward(self, input_dict):

    x, rec_targets, rec_lengths,radical_input,length_radical = input_dict['images'], \
                                  input_dict['rec_targets'], \
                                  input_dict['rec_lengths'], \
                                  input_dict['radical_input'], \
                                  input_dict['length_radical']


    # rectification
    # x = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
    if self.STN_ON:
      # input images are downsampled before being fed into stn_head.
      stn_input = F.interpolate(x, self.tps_inputsize, mode='bilinear', align_corners=True)
      stn_img_feat, ctrl_points = self.stn_head(stn_input)
      x, _ = self.tps(x, ctrl_points)
      if not self.training:
        # save for visualization
        return_dict['output']['ctrl_points'] = ctrl_points
        return_dict['output']['rectified_images'] = x

    encoder_feats,radical_feats = self.encoder(x)

    encoder_feats = encoder_feats.contiguous()
    embedding_vectors = self.embeder(encoder_feats)
    if self.training:
      rec_pred,att_map = self.decoder([encoder_feats, rec_targets, rec_lengths,input_dict['radical_input'].shape[1]],embedding_vectors)
    else:
      rec_pred,att_map = self.decoder([encoder_feats, rec_targets, rec_lengths,input_dict['radical_input'].shape[1]],embedding_vectors)

    #部首预测分支
    radical_decoder_result=self.radical_decoder(att_map.cuda(),radical_feats.cuda(),radical_input.cuda(),length_radical)


    return rec_pred,att_map,radical_decoder_result,embedding_vectors