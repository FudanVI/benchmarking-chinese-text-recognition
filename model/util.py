import torch
from data.lmdbReader import lmdbDataset, resizeNormalize
import os
import shutil
from shutil import copyfile
from torch.utils.data import Dataset
from zhconv import convert

#----------alphabet----------
def get_alp2num(args):
    alphabet_character_file = open(args.alpha_path)
    alphabet_character = list(alphabet_character_file.read().strip())
    alphabet_character_raw = ['START']
    for item in alphabet_character:
        alphabet_character_raw.append(item)
    alphabet_character_raw.append('END')
    alphabet_character = alphabet_character_raw

    alp2num_character = {}
    for index, char in enumerate(alphabet_character):
        alp2num_character[char] = index
    return alp2num_character


def get_dataloader(root, args, shuffle=False):
    dataset = lmdbDataset(root, resizeNormalize((args.imageW, args.imageH)))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch, shuffle=shuffle, num_workers=8,
    )
    return dataloader, dataset

def get_data_package(args):
    if not args.test_only:
        train_dataset = []
        for dataset_root in args.train_dataset.split(','):
            _, dataset = get_dataloader(dataset_root, args, shuffle=True)
            train_dataset.append(dataset)
        train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset_total, batch_size=args.batch, shuffle=True, num_workers=8,
        )
    test_dataset = []
    for dataset_root in args.test_dataset.split(','):
        _ , dataset = get_dataloader(dataset_root, args, shuffle=True)
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)

    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=args.batch, shuffle=False, num_workers=8,
    )

    if not args.test_only:
        return train_dataloader, test_dataloader
    else:
        return None, test_dataloader


def converter(label, args):
    "Convert string label to tensor"

    string_label = label
    label = [i for i in label]
    alp2num = get_alp2num(args)

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

    return length, text_input, text_all, string_label

def get_alphabet(args):
    alphabet_character_file = open(args.alpha_path)
    alphabet_character = list(alphabet_character_file.read().strip())
    alphabet_character_raw = ['START']
    for item in alphabet_character:
        alphabet_character_raw.append(item)
    alphabet_character_raw.append('END')
    alphabet_character = alphabet_character_raw
    return alphabet_character

def tensor2str(tensor, args):
    alphabet = get_alphabet(args)
    string = ""
    for i in tensor:
        if i == (len(alphabet)-1):
            continue
        string += alphabet[i]
    return string

def strQ2B(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring

def equal(pred, gt):
    pred = convert(strQ2B(pred.lower().replace(' ','')), 'zh-hans')
    gt = convert(strQ2B(gt.lower().replace(' ','')), 'zh-hans')
    if pred == gt:
        return True
    else:
        return False

def saver(args):
    try:
        shutil.rmtree('./history/{}'.format(args.exp_name))
    except:
        pass
    os.mkdir('./history/{}'.format(args.exp_name))

    src = './train.py'
    dst = os.path.join('./history', args.exp_name, 'train.py')
    copyfile(src, dst)

    src = './util.py'
    dst = os.path.join('./history', args.exp_name, 'util.py')
    copyfile(src, dst)

    src = './args.py'
    dst = os.path.join('./history', args.exp_name, 'args.py')
    copyfile(src, dst)

    src = './model/TransformerSTR.py'
    dst = os.path.join('./history', args.exp_name, 'TransformerSTR.py')
    copyfile(src, dst)

    src = './model/TransformerUtil.py'
    dst = os.path.join('./history', args.exp_name, 'TransformerUtil.py')
    copyfile(src, dst)

    src = './model/ResNet.py'
    dst = os.path.join('./history', args.exp_name, 'ResNet.py')
    copyfile(src, dst)