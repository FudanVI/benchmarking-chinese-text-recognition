from __future__ import print_function
from collections import OrderedDict
from torch.autograd import Variable
import argparse
import os
import random
import time
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import zhconv

from model.moran import MORAN
import dataset
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', help='activate to train a model')
parser.add_argument('--test', action='store_true', help='activate to test a trained model')
parser.add_argument('--trainroot', default='', help='path to train dataset')
parser.add_argument('--valroot', default='', help='path to val dataset')
parser.add_argument('--testroot', default='', help='path to test dataset')
parser.add_argument('--worker', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batch', type=int, default=48, help='input batch size')
parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
parser.add_argument('--imgW', type=int, default=256, help='the width of the input image to network')
parser.add_argument('--targetH', type=int, default=32, help='the width of the input image to network')
parser.add_argument('--targetW', type=int, default=256, help='the width of the input image to network')
parser.add_argument('--nh', type=int, default=256, help='size of the lstm hidden state')
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=1, help='learning rate for Critic, default=1')
parser.add_argument('--handwriting', action='store_true',
                    help='activate when training handwriting dataset to apply a specific lr decay policy')
parser.add_argument('--cuda', default=True, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--weight', default='', help="path to model (to continue training)")
parser.add_argument('--alphabet', type=str, default='data/alphabet.txt', help="path to alphabet")
parser.add_argument('--experiment', default='expr', help='Where to store samples and models')
parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
parser.add_argument('--optimizer', default='adadelta', help='optimizer to use')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--radical', type=float, default=0, help='weight of radical loss, stay 0 for baseline')
parser.add_argument('--manualSeed', type=int, default=1234, help='manual seed')

opt = parser.parse_args()
print(opt)


# load alphabet
alphabet_character_file = open(opt.alphabet)
alphabet_character = list(alphabet_character_file.read().strip())
alphabet_character_raw = ['START', '\xad']

for item in alphabet_character:
    alphabet_character_raw.append(item)

alphabet_character_raw.append('END')
alphabet_character = alphabet_character_raw

alp2num_character = {}
# alp2num_character['<']=0
# alp2num_character['\xad']=1

for index, char in enumerate(alphabet_character):
    alp2num_character[char] = index

# load radical lexicon
alphabet_radical_file = open("data/radicals.txt")
alphabet_radical = alphabet_radical_file.readlines()
alphabet_radical = [item.strip('\n') for item in alphabet_radical]
alphabet_radical_raw = ['START']

for item in alphabet_radical:
    alphabet_radical_raw.append(item)

alphabet_radical_raw.append('END')
alphabet_radical = alphabet_radical_raw

alp2num_radical = {}

for index, char in enumerate(alphabet_radical):
    alp2num_radical[char] = index

# load decomposition dict
files = open("data/decompose.txt").readlines()
char2radicals = {}
for line in files:
    items = line.strip('\n').strip().split(':')
    char2radicals[items[0]] = items[1].split(' ')

for i in range(1, len(alphabet_character)-1):
    if alphabet_character[i] not in char2radicals.keys():
        char2radicals[alphabet_character[i]] = alphabet_character[i]


# convert text label to tensor
def converter_radical(label):

    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            text_input[i][j + 1] = alp2num[label[i][j]]

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i])-1):
                text_all[start + j] = alp2num['END']
            else:
                text_all[start + j] = alp2num[label[i][j]]
        start += len(label[i])

    length_radical = []
    for i in range(batch):
        length_tmp = []
        for j in range(max_length):
            if j < len(label[i])-1:
                length_tmp.append(len(char2radicals[label[i][j]])+1)
            else:
                length_tmp.append(0)
        length_radical.append(length_tmp)
    length_radical = torch.Tensor(length_radical).long().cuda()

    max_radical_len = max(length_radical.view(-1))
    radical_input = torch.zeros(batch, max_length, max_radical_len).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            radicals = char2radicals[label[i][j]]
            for k in range(len(radicals)):
                radical_input[i][j][k+1] = k + 1

    sum_length = sum(length_radical.view(-1))
    radical_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])-1):
            radicals = char2radicals[label[i][j]]
            for k in range(len(radicals)):
                try:
                    radical_all[start + k] = alp2num_radical[radicals[k]]
                except KeyError:
                    alp2num_radical[radicals[k]] = len(alp2num_radical)
                    radical_all[start + k] = alp2num_radical[radicals[k]]
            radical_all[start + len(radicals)] = alp2num_radical['END']
            start += (len(radicals) + 1)

    return length, text_input, text_all, length_radical, radical_input, radical_all, string_label

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


f = open(opt.alphabet, 'rb')
voc_list = f.read()
f.close()
alphabet = voc_list.decode('utf-8')

assert opt.ngpu == 1, "Multi-GPU training is not supported yet, due to the variant lengths of the text in a batch."

os.system('mkdir {0}'.format(opt.experiment))

random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if not torch.cuda.is_available():
    assert not opt.cuda, 'You don\'t have a CUDA device.'

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# compatibility modification
if opt.weight != '':
    if opt.cuda:
        state_dict = torch.load(opt.weight)
    else:
        state_dict = torch.load(opt.weight, map_location='cpu')
    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        MORAN_state_dict_rename[name] = v
    nclass = MORAN_state_dict_rename['ASRN.attention.generator.bias'].shape[0]
    if nclass == 4406:
        f = open('data/web.txt', 'rb')
        voc_list = f.read()
        f.close()
        alphabet = voc_list.decode('utf-8')
    elif nclass == 5899:
        f = open('data/scene.txt', 'rb')
        voc_list = f.read()
        f.close()
        alphabet = voc_list.decode('utf-8')
    elif nclass == 4869:
        f = open('data/document.txt', 'rb')
        voc_list = f.read()
        f.close()
        alphabet = voc_list.decode('utf-8')

radical_state_dicts = {
    "ASRN.radical_branch.attention_compress.weight": torch.zeros(1, 4), 
    "ASRN.radical_branch.attention_compress.bias": torch.zeros(1), 
    "ASRN.radical_branch.features_compress.weight": torch.zeros(64, 512, 1, 1), 
    "ASRN.radical_branch.features_compress.bias": torch.zeros(64), 
    "ASRN.radical_branch.embedding_radical.lut.weight": torch.zeros(nclass, 256), 
    "ASRN.radical_branch.pe_radical.pe": torch.zeros(1, 8000, 256), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.linears.0.weight": torch.zeros(512, 512), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.linears.0.bias": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.linears.1.weight": torch.zeros(512, 512), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.linears.1.bias": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.linears.2.weight": torch.zeros(512, 512), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.linears.2.bias": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.linears.3.weight": torch.zeros(512, 512), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.linears.3.bias": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.compress_attention_linear.weight": torch.zeros(1, 4), 
    "ASRN.radical_branch.decoder_radical.mask_multihead.compress_attention_linear.bias": torch.zeros(1), 
    "ASRN.radical_branch.decoder_radical.mul_layernorm1.a_2": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.mul_layernorm1.b_2": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.multihead.linears.0.weight": torch.zeros(512, 512), 
    "ASRN.radical_branch.decoder_radical.multihead.linears.0.bias": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.multihead.linears.1.weight": torch.zeros(512, 512), 
    "ASRN.radical_branch.decoder_radical.multihead.linears.1.bias": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.multihead.linears.2.weight": torch.zeros(512, 512), 
    "ASRN.radical_branch.decoder_radical.multihead.linears.2.bias": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.multihead.linears.3.weight": torch.zeros(512, 512), 
    "ASRN.radical_branch.decoder_radical.multihead.linears.3.bias": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.multihead.compress_attention_linear.weight": torch.zeros(1, 4), 
    "ASRN.radical_branch.decoder_radical.multihead.compress_attention_linear.bias": torch.zeros(1), 
    "ASRN.radical_branch.decoder_radical.mul_layernorm2.a_2": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.mul_layernorm2.b_2": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.pff.w_1.weight": torch.zeros(1024, 512), 
    "ASRN.radical_branch.decoder_radical.pff.w_1.bias": torch.zeros(1024), 
    "ASRN.radical_branch.decoder_radical.pff.w_2.weight": torch.zeros(512, 1024), 
    "ASRN.radical_branch.decoder_radical.pff.w_2.bias": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.mul_layernorm3.a_2": torch.zeros(512), 
    "ASRN.radical_branch.decoder_radical.mul_layernorm3.b_2": torch.zeros(512), 
    "ASRN.radical_branch.generator_radical.proj.weight": torch.zeros(nclass, 512), 
    "ASRN.radical_branch.generator_radical.proj.bias": torch.zeros(nclass)
    }


if opt.train:
    train_dataset = dataset.lmdbDataset(root=opt.trainroot, 
        transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=False, alphabet=alphabet)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch,
        shuffle=False, sampler=dataset.randomSequentialSampler(train_dataset, opt.batch),
        num_workers=int(opt.worker))
    
    val_dataset = dataset.lmdbDataset(root=opt.valroot, 
        transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=False, alphabet=alphabet)

if opt.test:
    test_dataset = dataset.lmdbDataset(root=opt.testroot, 
        transform=dataset.resizeNormalize((opt.imgW, opt.imgH)), reverse=False, alphabet=alphabet)

nclass = len(alphabet)
nc = 1

converter = utils.strLabelConverterForAttention(list(alphabet))
criterion = torch.nn.CrossEntropyLoss()

if opt.cuda:
    MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, CUDA=opt.cuda)
else:
    MORAN = MORAN(nc, nclass, opt.nh, opt.targetH, opt.targetW, inputDataType='torch.FloatTensor', CUDA=opt.cuda)

total_params = sum(p.numel() for p in MORAN.parameters())
print("Total parameter of the model is:", total_params)

# load chechpoint
if opt.weight != '':
    print('loading pretrained model from %s' % opt.weight)
    if opt.cuda:
        state_dict = torch.load(opt.weight)
    else:
        state_dict = torch.load(opt.weight, map_location='cpu')
    MORAN_state_dict_rename = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        MORAN_state_dict_rename[name] = v
    if opt.test or opt.radical == 0:
        for key in radical_state_dicts:
            MORAN_state_dict_rename[key] = radical_state_dicts[key]
    MORAN.load_state_dict(MORAN_state_dict_rename, strict=True)

image = torch.FloatTensor(opt.batch, nc, opt.imgH, opt.imgW)
text = torch.LongTensor(opt.batch * 5)
text_rev = torch.LongTensor(opt.batch * 5)
length = torch.IntTensor(opt.batch)

if opt.cuda:
    MORAN.cuda()
    MORAN = torch.nn.DataParallel(MORAN, device_ids=range(opt.ngpu))
    image = image.cuda()
    text = text.cuda()
    text_rev = text_rev.cuda()
    criterion = criterion.cuda()

image = Variable(image)
text = Variable(text)
text_rev = Variable(text_rev)
length = Variable(length)

# loss averager
loss_avg = utils.averager()

# setup optimizer
if opt.optimizer == 'adam':
    optimizer = optim.Adam(MORAN.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
elif opt.optimizer == 'adadelta':
    optimizer = optim.Adadelta(MORAN.parameters(), lr=opt.lr)
elif opt.optimizer == 'sgd':
    optimizer = optim.SGD(MORAN.parameters(), lr=opt.lr, momentum=0.9)
else:
    optimizer = optim.RMSprop(MORAN.parameters(), lr=opt.lr)

def val(dataset, criterion):
    print('Start val')
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batch, num_workers=int(opt.worker))
    val_iter = iter(data_loader)
    max_iter = len(data_loader)
    n_correct = 0
    n_total = 0
    loss_avg = utils.averager()
    
    for i in range(max_iter):
        data = val_iter.next()
        cpu_images, cpu_texts = data
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts, scanned=True)
        utils.loadData(text, t)
        utils.loadData(length, l)
        preds, _ = MORAN(image, length, text, text_rev, test=True)
        cost = criterion(preds, text)
        _, preds = preds.max(1)
        preds = preds.view(-1)
        sim_preds = converter.decode(preds.data, length.data)

        loss_avg.add(cost)
        sim_preds = [zhconv.convert(strQ2B(pred), 'zh-cn') for pred in sim_preds]
        cpu_texts = [zhconv.convert(strQ2B(tar), 'zh-cn')for tar in cpu_texts]
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1
            n_total += 1
    
    accuracy = n_correct / float(n_total)
    print("correct / total: %d / %d, Test loss: %f, accuray: %f"  % (n_correct, n_total, loss_avg.val(), accuracy))
    return accuracy

def trainBatch():
    data = train_iter.next()
    
    if opt.radical != 0:
        coeff = opt.radical
        cpu_images, cpu_texts = data
        _, _, _, length_radical, radical_input, radical_gt, _ = converter_radical(cpu_texts)
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts, scanned=True)
        utils.loadData(text, t)
        utils.loadData(length, l)
        preds, radical_preds = MORAN(image, length, text, text_rev, length_radical, radical_input)
        text_cost = criterion(preds, text)
        radical_cost = criterion(radical_preds['radical_pred'], radical_gt)
        cost = text_cost + coeff * radical_cost
    else:
        cpu_images, cpu_texts = data
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts, scanned=True)
        utils.loadData(text, t)
        utils.loadData(length, l)
        preds, _ = MORAN(image, length, text, text_rev)
        cost = criterion(preds, text)

    MORAN.zero_grad()
    cost.backward()
    optimizer.step()
    return cost

def test(dataset):
    print('Start test')
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=opt.batch, num_workers=int(opt.worker))
    val_iter = iter(data_loader)
    max_iter = len(data_loader)
    n_correct = 0
    n_total = 0
    test_result = []
    
    for i in range(max_iter):
        data = val_iter.next()
        cpu_images, cpu_texts = data
        utils.loadData(image, cpu_images)
        t, l = converter.encode(cpu_texts, scanned=True)
        utils.loadData(text, t)
        utils.loadData(length, l)
        preds, _ = MORAN(image, length, text, text_rev, test=True)
        _, preds = preds.max(1)
        preds = preds.view(-1)
        sim_preds = converter.decode(preds.data, length.data)
        sim_preds = [zhconv.convert(strQ2B(pred), 'zh-cn') for pred in sim_preds]
        cpu_texts = [zhconv.convert(strQ2B(tar), 'zh-cn')for tar in cpu_texts]

        for pred, target in zip(sim_preds, cpu_texts):
            result = (str(n_total) + ' | ' + pred + ' | ' + target).replace('$', '')
            test_result.append(result)
            print(result)
            if pred == target.lower():
                n_correct += 1
            n_total += 1
    
    f = open(opt.experiment + '/test_result.txt', 'w')
    for line in test_result:
        f.write(line + '\n')
    f.close()
    
    accuracy = n_correct / float(n_total)
    print("correct / total: %d / %d, accuray: %f"  % (n_correct, n_total, accuracy))
    return accuracy

if opt.train:
    t0 = time.time()
    acc = 0
    acc_tmp = 0
    for epoch in range(opt.niter):
    
        train_iter = iter(train_loader)
        i = 0
        for p in MORAN.parameters():
            p.requires_grad = True
        while i < len(train_loader):
            i += 1
            try:
                MORAN.train()
                cost = trainBatch()
                loss_avg.add(cost)
                
                if i % opt.displayInterval == 0:
                    t1 = time.time()            
                    print ('Epoch: %d/%d; iter: %d/%d; Loss: %f; time: %.2f s;' %
                            (epoch, opt.niter, i, len(train_loader), loss_avg.val(), t1-t0)),
                    loss_avg.reset()
                    t0 = time.time()
                torch.cuda.empty_cache()
            
            except:
                print('Epoch: %d/%d; iter: %d/%d' %
                            (epoch, opt.niter, i, len(train_loader)), " is skipped due to memory explosion;")
                torch.cuda.empty_cache()
                continue
        
        for p in MORAN.parameters():
            p.requires_grad = False
        MORAN.eval()
    
        acc_tmp = val(val_dataset, criterion)
        if acc_tmp > acc:
            acc = acc_tmp
            torch.save(MORAN.state_dict(), '{0}/model_best_{1}.pth'.format(opt.experiment, str(acc)[:7]))
        
        # torch.save(MORAN.state_dict(), '{0}/model_{1}.pth'.format(opt.experiment, epoch))
        if not opt.handwriting:
            if (epoch + 1) <= 12 and (epoch + 1) % 4 == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.8
            elif (epoch + 1) > 12 and (epoch + 1) <= 20 and (epoch + 1) % 2 == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.8
            elif (epoch + 1) > 20:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.8
        else:
            if (epoch + 1) % 15 == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.8

if opt.test:
    for p in MORAN.parameters():
        p.requires_grad = False
    MORAN.eval()
    acc_tmp = test(test_dataset)