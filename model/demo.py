import torch
import torch.nn as nn
import argparse

from PIL import Image

from model.TransformerSTR import Transformer
from util import tensor2str, get_alphabet
from data.lmdbReader import resizeNormalize

parser = argparse.ArgumentParser()
parser.add_argument('--resume', default='', help="path to pretrained model (to continue training)")
parser.add_argument('--imageH', type=int, default=64, help='the height of the input image to network')
parser.add_argument('--imageW', type=int, default=200, help='the width of the input image to network')
parser.add_argument('--alpha_path', default='', help='path to alphabet')
parser.add_argument('--max_len', type=int, default=100, help='the max length of model prediction')
parser.add_argument('--image_path', default='', help='path to image')
parser.add_argument('--dataset', type=str, required=True, choices=['Web', 'Scene', 'Document', 'Handwriting'], help='the type of dataset')

args = parser.parse_args()

#get the alphabet
alphabet = get_alphabet(args)

# build the model
model = Transformer(args).cuda()
model = nn.DataParallel(model)
#--------load pretrain model-------
if args.resume.strip() != '':
    print('loading model..')
    model.load_state_dict(torch.load(args.resume))

model.eval()

# load the image
transformer = resizeNormalize((args.imageW, args.imageH))
image = Image.open(args.image_path).convert('RGB')
image = transformer(image)
image = image.unsqueeze(0).cuda()

max_length = args.max_len
pred = torch.zeros(1, 1).long().cuda()
image_features = None

for i in range(max_length):
    length_tmp = torch.zeros(1).long().cuda() + i + 1
    result = model(image, length_tmp, pred, conv_feature=image_features, test=True)
    prediction = result['pred']
    now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
    pred = torch.cat((pred, now_pred[:, -1].view(-1, 1)), 1)
    image_features = result['conv']

pred = pred[0]
pred_list = []
for j in range(max_length):
    if pred[j] != len(alphabet) - 1:
        pred_list.append(pred[j])
    else:
        break
pred_list = torch.Tensor(pred_list)[1:].long().cuda()
pred = tensor2str(pred_list, args)
print('prediction:', pred)
