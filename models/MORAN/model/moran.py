import torch.nn as nn
from model.morn import MORN
from model.asrn_res import ASRN

class MORAN(nn.Module):

    def __init__(self, nc, nclass, nh, targetH, targetW, BidirDecoder=False, 
    	inputDataType='torch.cuda.FloatTensor', maxBatch=256, CUDA=True):
        super(MORAN, self).__init__()
        self.MORN = MORN(nc, targetH, targetW, inputDataType, maxBatch, CUDA)
        self.ASRN = ASRN(targetH, nc, nclass, nh, BidirDecoder, CUDA)

    def forward(self, x, length, text, text_rev, length_radical=None, radical_input=None, test=False, debug=False):
        if debug:
            x_rectified, demo = self.MORN(x, test, debug=debug)
            preds = self.ASRN(x_rectified, length, text, text_rev, test)
            return preds, demo
        else:
            x_rectified = self.MORN(x, test, debug=debug)
            preds, radical = self.ASRN(x_rectified, length, text, text_rev, length_radical, radical_input, test)
            return preds, radical
