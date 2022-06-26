from collections import OrderedDict
from torch.multiprocessing import freeze_support
import argparse
import math
import os
import random
import time
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data

from model.sar import sar
from utils.dataproc import performance_evaluate
import dataset
import ut


# load alphabet
alphabet_character_file = open("data/alphabet.txt")
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
def converter(label, max_length):

    string_label = label
    label = [i for i in label]
    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()

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


def dictionary_generator(alphabet, END='END', PADDING='PAD', UNKNOWN='UNK'):
    '''
    END: end of sentence token
    PADDING: padding token
    UNKNOWN: unknown character token
    '''
    
    voc = list(alphabet)
    
    # update the voc with 3 specifical chars
    voc.append(END)
    voc.append(PADDING)
    voc.append(UNKNOWN)

    char2id = dict(zip(voc, range(len(voc))))
    id2char = dict(zip(range(len(voc)), voc))

    return voc, char2id, id2char

# main function:
if __name__ == '__main__':
    
    freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='activate to train a model')
    parser.add_argument('--test', action='store_true', help='activate to test a trained model')
    parser.add_argument('--trainroot', default='', help='path to train dataset')
    parser.add_argument('--valroot', default='', help='path to val dataset')
    parser.add_argument('--testroot', default='', help='path to test dataset')
    parser.add_argument('--worker', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch', type=int, default=64, help='input batch size')
    parser.add_argument('--imgH', type=int, default=32, help='the height of the input image to network')
    parser.add_argument('--imgW', type=int, default=256, help='the width of the input image to network')
    parser.add_argument('--niter', type=int, default=80, help='number of epochs')
    parser.add_argument('--document', action='store_true',
                    help='activate when training and testing document dataset to set input channels to 1')
    parser.add_argument('--gpu', type=bool, default=True, help="GPU being used or not")
    parser.add_argument('--metric', type=str, default='accuracy', help="evaluation metric - accuracy|editdistance")
    parser.add_argument('--maxlen', type=int, default=50, help='max sequence length of label')
    parser.add_argument('--weight', type=str, default='', help='model path')
    parser.add_argument('--alphabet', type=str, default='data/alphabet.txt', help="path to alphabet")
    parser.add_argument('--experiment', type=str, default='expr', help='output folder name')
    parser.add_argument('--displayInterval', type=int, default=100, help='Interval to be displayed')
    parser.add_argument('--radical', type=float, default=0, help='weight of radical loss, stay 0 for baseline')
    parser.add_argument('--manualSeed', type=int, default=1234, help='manual seed')
    
    opt = parser.parse_args()
    print(opt)
    
    f = open(opt.alphabet, 'rb')
    voc_list = f.read()
    f.close()
    alphabet = voc_list.decode('utf-8')
    
    # compatibility modification
    if opt.weight != '':
        if opt.gpu:
            state_dict = torch.load(opt.weight)
        else:
            state_dict = torch.load(opt.weight, map_location='cpu')
        state_dict_rename = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("module.", "")
            state_dict_rename[name] = v
        nclass = state_dict_rename['decoder_model.linear2.bias'].shape[0]
        if nclass == 4409:
            f = open('data/web.txt', 'rb')
            voc_list = f.read()
            f.close()
            alphabet = voc_list.decode('utf-8')
        elif nclass == 5902:
            f = open('data/scene.txt', 'rb')
            voc_list = f.read()
            f.close()
            alphabet = voc_list.decode('utf-8')
        elif nclass == 4872:
            f = open('data/document.txt', 'rb')
            voc_list = f.read()
            f.close()
            alphabet = voc_list.decode('utf-8')
    
    converter_sar = ut.strLabelConverter(alphabet)
    
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    
    # turn on GPU for models:
    if opt.gpu == False:
        device = torch.device("cpu")
        print("CPU being used!")
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            device = torch.device("cuda")
            print("GPU being used!")
        else:
            device = torch.device("cpu")
            print("CPU being used!")
    
    # set training parameters
    batch_size = opt.batch
    Height = opt.imgH
    Width = opt.imgW
    feature_height = Height // 4
    feature_width = Width // 8
    Channel = 3
    if opt.document:
        Channel = 1
    voc, char2id, id2char = dictionary_generator(alphabet)
    output_classes = len(voc)
    print("Num of output classes is:", output_classes)
    embedding_dim = 512
    hidden_units = 512
    layers = 2
    keep_prob = 1.0
    coeff = opt.radical
    seq_len = opt.maxlen
    epochs = opt.niter
    worker = opt.worker
    train_path = opt.trainroot
    val_path = opt.valroot
    test_path = opt.testroot
    output_path = opt.experiment
    trained_model_path = opt.weight
    eval_metric = opt.metric
    
    # create dataset
    print("Create dataset......")
    if opt.train:
        train_dataset = dataset.lmdbDataset(seq_len, output_classes, voc, char2id, root=train_path, radical=coeff)
        val_dataset = dataset.lmdbDataset(seq_len, output_classes, voc, char2id, root=val_path)
    if opt.test:
        test_dataset = dataset.lmdbDataset(seq_len, output_classes, voc, char2id, root=test_path)
    
    # make dataloader
    if opt.train:
        train_dataloader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=int(worker),
                        collate_fn=dataset.alignCollate(imgH=Height, imgW=Width))
            
        val_dataloader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=int(worker),
                        collate_fn=dataset.alignCollate(imgH=Height, imgW=Width))
        print("Length of train dataset is:", len(train_dataset))
        print("Length of val dataset is:", len(val_dataset))
    
    if opt.test:
        test_dataloader = torch.utils.data.DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=int(worker),
                        collate_fn=dataset.alignCollate(imgH=Height, imgW=Width))
        print("Length of test dataset is:", len(test_dataset))
    
    # make model output folder
    try:
        os.makedirs(output_path)
    except OSError:
        pass

    # create model
    print("Create model......")
    model = sar(Channel, feature_height, feature_width, embedding_dim, output_classes, hidden_units, layers, keep_prob, seq_len, device)
    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameter of the model is:", total_params)

    if trained_model_path != '':
        if torch.cuda.is_available() == True and opt.gpu == True:
            model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
            model = torch.nn.DataParallel(model).to(device)
        else:
            model.load_state_dict(torch.load(trained_model_path, map_location=lambda storage, loc: storage), strict=False)
    else:
        if torch.cuda.is_available() == True and opt.gpu == True:
            model = torch.nn.DataParallel(model).to(device)
        else:
            model = model.to(device)

    image = torch.FloatTensor(batch_size, Channel, Height, Width)
    image = image.cuda()
    text = torch.FloatTensor(batch_size, seq_len, output_classes)
    text = text.cuda()
    image = torch.autograd.Variable(image)
    text = torch.autograd.Variable(text)
    
    # train, evaluate, and save model
    if opt.train:
        
        optimizer = optim.Adadelta(model.parameters())
        num_batch = math.ceil(len(train_dataset) / batch_size)
        radical_criterion = torch.nn.CrossEntropyLoss().cuda()
        
        print("Training starts......")
        if eval_metric == 'accuracy':
            best_acc = float('-inf')
        elif eval_metric == 'editdistance':
            best_acc = float('inf')
        else:
            print("Wrong --metric argument, set it to default")
            eval_metric = 'accuracy'
            best_acc = float('-inf')
    
        for epoch in range(epochs):
            M_list = []
            train_loader = iter(train_dataloader)
            for i in range(len(train_dataloader)):
                start_time = time.time()
                data = train_loader.next()
                if coeff != 0:
                    x, tps = data
                    y = [tp[0] for tp in tps]
                    t = torch.stack(y).squeeze(1)
                    label = [tp[1] for tp in tps]
                    _, _, _, length_radical, radical_input, radical_gt, _ = converter(label, seq_len)
                    ut.loadData(image, x)
                    ut.loadData(text, t)
                    optimizer.zero_grad()
                    model = model.train()
                    
                    predict, _, _, _, radical_pred = model(image, text, length_radical, radical_input)
                    target = text.max(2)[1] # [batch_size, seq_len]
                    predict_reshape = predict.permute(0,2,1) # [batch_size, output_classes, seq_len]
                    
                    loss_char = F.nll_loss(predict_reshape, target)
                    loss_radical = radical_criterion(radical_pred['radical_pred'], radical_gt)
                    
                    loss = loss_char + coeff * loss_radical
                else:
                    x, y = data
                    t = torch.stack(y).squeeze(1)
                    ut.loadData(image, x)
                    ut.loadData(text, t)
                    optimizer.zero_grad()
                    model = model.train()
                    predict, _, _, _, _ = model(image, text)
                    target = text.max(2)[1] # [batch_size, seq_len]
                    predict_reshape = predict.permute(0,2,1) # [batch_size, output_classes, seq_len]
                    loss = F.nll_loss(predict_reshape, target)
                loss.backward()
                optimizer.step()
                # prediction evaluation
                pred_choice = predict.max(2)[1] # [batch_size, seq_len]
                metric, metric_list, predict_words, labeled_words = performance_evaluate(pred_choice.detach().cpu().numpy(), target.detach().cpu().numpy(), voc, char2id, id2char, eval_metric)
                M_list += metric_list
                if i % opt.displayInterval == 0:
                    print('[Epoch %d: %d/%d] train loss: %f accuracy: %f' % (epoch, i, num_batch, loss.item(), metric) + " TIME: " + str(time.time()-start_time))
                torch.cuda.empty_cache()
            train_acc = float(sum(M_list)/len(M_list))
            print("Epoch {} average train accuracy: {}".format(epoch, train_acc))
            
            if (epoch + 1) <= 12 and (epoch + 1) % 4 == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.8
            elif (epoch + 1) > 12 and (epoch + 1) <= 20 and (epoch + 1) % 2 == 0:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.8
            elif (epoch + 1) > 20:
                for p in optimizer.param_groups:
                    p['lr'] *= 0.8
            
            # Validation
            print("Validating......")
            with torch.set_grad_enabled(False):
                M_list = []
                val_loader = iter(val_dataloader)
                for i in range(len(val_dataloader)):
                    data = val_loader.next()
                    x, y = data
                    t = torch.stack(y).squeeze(1)
                    ut.loadData(image, x)
                    ut.loadData(text, t)
                    model = model.eval()
                    predict, _, _, _, _ = model(image, text, test=True)
                    # prediction evaluation
                    pred_choice = predict.max(2)[1] # [batch_size, seq_len]
                    target = text.max(2)[1] # [batch_size, seq_len]
                    metric, metric_list, predict_words, labeled_words = performance_evaluate(pred_choice.detach().cpu().numpy(), target.detach().cpu().numpy(), voc, char2id, id2char, eval_metric)
                    M_list += metric_list
                test_acc = float(sum(M_list)/len(M_list))
                print("Epoch {} average test accuracy: {}".format(epoch, test_acc))
                with open(os.path.join(output_path,'statistics.txt'), 'a') as f:
                    f.write("{} {}\n".format(train_acc, test_acc))
                if eval_metric == 'accuracy':
                    if test_acc >= best_acc:
                        print("Save current best model with accuracy:", test_acc)
                        best_acc = test_acc
                        if torch.cuda.is_available() == True and opt.gpu == True:
                            torch.save(model.module.state_dict(), '%s/model_best.pth' % (output_path))
                        else:
                            torch.save(model.state_dict(), '%s/model_best.pth' % (output_path))
                elif eval_metric == 'editdistance':
                    if test_acc <= best_acc:
                        print("Save current best model with accuracy:", test_acc)
                        best_acc = test_acc
                        if torch.cuda.is_available() == True and opt.gpu == True:
                            torch.save(model.module.state_dict(), '%s/model_best.pth' % (output_path))
                        else:
                            torch.save(model.state_dict(), '%s/model_best.pth' % (output_path))
        print("Best test accuracy is:", best_acc)

    # test
    if opt.test:
        print("Testing......")
        test_result = []
        M_list = []
        with torch.set_grad_enabled(False):
            count = 0
            test_loader = iter(test_dataloader)
            for i in range(len(test_dataloader)):
                data = test_loader.next()
                x, y = data
                t = torch.stack(y).squeeze(1)
                ut.loadData(image, x)
                ut.loadData(text, t)
                model = model.eval()
                predict, _, _, _, _ = model(image, text, test=True)
                # prediction evaluation
                pred_choice = predict.max(2)[1] # [batch_size, seq_len]
                target = text.max(2)[1] # [batch_size, seq_len]
                metric, metric_list, predict_words, labeled_words = performance_evaluate(pred_choice.detach().cpu().numpy(), target.detach().cpu().numpy(), voc, char2id, id2char, eval_metric)
                
                for j in range(len(predict_words)):
                    result = str(count) + ' | ' + predict_words[j] + ' | ' + labeled_words[j]
                    test_result.append(result)
                    print(result)
                    count += 1
                
                M_list += metric_list
            
            test_acc = float(sum(M_list)/len(M_list))
        
        f = open(output_path + 'test_result.txt', 'w')
        for line in test_result:
            f.write(line + '\n')
        f.close()
        
        print("Best test accuracy is:", test_acc)