'''
The code for postprocessing.
'''
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

def attention_map(predict_word, x, attention_weight):
    '''
    Input:
    predict_word: string of predicted word
    x: tensor of original image [C, H, W], channel in BGR order, normalized to [-1, 1]
    attention_weight: tensor of attention weights [seq_len, 1, feature_H, feature_W] in [0, 1]
    Output:
    heatmaps: a list of heatmap [H, W, C=3]
    overlaps: a list of overlapped image [H, W, C=3]
    '''
    T = len(predict_word)
    x = x.permute(1,2,0).detach().cpu().numpy() # [H, W, C]
    x = (((x+1)/2)*255).astype(np.uint8) # normalized to [0,255]
    H, W, C = x.shape
    heatmaps = []
    overlaps = []
    for t in range(T):
        att_map = attention_weight[t,:,:,:].permute(1,2,0).detach().cpu().numpy() # [feature_H, feature_W, 1]
        att_map = cv2.resize(att_map, (W,H)) # [H, W]
        att_map = (att_map*255).astype(np.uint8)
        heatmap = cv2.applyColorMap(att_map, cv2.COLORMAP_JET) # [H, W, C]
        overlap = cv2.addWeighted(heatmap, 0.6, x, 0.4, 0)
        heatmaps.append(heatmap)
        overlaps.append(overlap)

    return heatmaps, overlaps

# unit test
if __name__ == '__main__':

    img_path = '../svt/img/00_16.jpg'

    predict_word = "hello"

    x = cv2.imread(img_path)

    x = (x-127.5)/127.5 # normalization

    x = torch.from_numpy(x)

    x = x.permute(2, 0, 1) # [C, H, W]

    attention_weight = torch.rand((40, 1, 384, 512))

    attention_weight[:,:,250:300,150:200] = 1.0

    attention_weight[:,:,0:50,0:50] = 0.0

    attention_weight[:,:,300:350,450:500] = 0.5

    heatmaps, overlaps = attention_map(predict_word, x, attention_weight)

    heatmap_single = cv2.cvtColor(heatmaps[-1], cv2.COLOR_BGR2RGB)
    overlap_single = cv2.cvtColor(overlaps[-1], cv2.COLOR_BGR2RGB)

    plt.figure(0)
    plt.imshow(heatmap_single)

    plt.figure(1)
    plt.imshow(overlap_single)

    plt.show()