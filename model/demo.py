import torch
import torch.nn as nn

from PIL import Image

from args import args
from model.TransformerSTR import Transformer
from util import tensor2str, get_alphabet
from data.lmdbReader import resizeNormalize

#get the alphabet
alphabet = get_alphabet(args)

# build the model
model = Transformer(args).cuda()
model = nn.DataParallel(model)
model.load_state_dict(torch.load(args.resume))
model.eval()

# load the image
transformer = resizeNormalize((args.imageW, args.imageH))
image = Image.open(args.image_path).convert('L')
image = transformer(image)

max_length = args.max_len
pred = torch.zeros(1, 1).long().cuda()
image_features = None
prob = torch.zeros(1, max_length).float()

for i in range(max_length):
    length_tmp = torch.zeros(1).long().cuda() + i + 1
    result = model(image, length_tmp, pred, conv_feature=image_features, test=True)
    prediction = result['pred']
    now_pred = torch.max(torch.softmax(prediction, 2), 2)[1]
    prob[:, i] = torch.max(torch.softmax(prediction, 2), 2)[0][:, -1]
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
pred = tensor2str(pred_list)
print(pred)
