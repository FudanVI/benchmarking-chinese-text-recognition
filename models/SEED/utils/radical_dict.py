#
import torch
import os
import shutil
from shutil import copyfile
import pickle as pkl
import torch.nn as nn
from torch.utils.data import Dataset

# 获取训练和测试的字符集
alphabet_character_file = open("./data/benchmark_new.txt")
alphabet_character = list(alphabet_character_file.read().strip())
alphabet_character_raw = ['PADDING','UNKNOW', '\xad']

for item in alphabet_character:
    alphabet_character_raw.append(item)

alphabet_character_raw.append('END')
alphabet_character = alphabet_character_raw

alp2num_character = {}
num2alp_character = {}
# alp2num_character['<']=0
# alp2num_character['\xad']=1

for index, char in enumerate(alphabet_character):
    alp2num_character[char] = index
    num2alp_character[index] = char


# 获取部首字符表
alphabet_radical_file = open("./data/radicals.txt")
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
# print("alp2num_radical",alp2num_radical)
# 获取字符到部首序列的dict
files = open("./data/decompose.txt").readlines()
char2radicals = {}
for line in files:
    items = line.strip('\n').strip().split(':')
    char2radicals[items[0]] = items[1].split(' ')

for i in range(1, len(alphabet_character) - 1):
    if alphabet_character[i] not in char2radicals.keys():
        char2radicals[alphabet_character[i]] = alphabet_character[i]


# # 由label转变成可以输入模型的tensor
def converter(label):
    # label的长度是包含结束符的
    string_label = label
    label = [i for i in label]

    alp2num = alp2num_character

    batch = len(label)
    length = torch.Tensor([len(i) for i in label]).long().cuda()  # 字符包含的部首长度 含$
    max_length = max(length)

    text_input = torch.zeros(batch, max_length).long().cuda()
    for i in range(batch):
        for j in range(len(label[i])):
            text_input[i][j] = alp2num[label[i][j].lower()]
            # text_input[i][j + 1] = j + 1

    sum_length = sum(length)
    text_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i])):
            if j == (len(label[i]) - 1):
                text_all[start + j] = alp2num['END']
            else:
                text_all[start + j] = alp2num[label[i][j].lower()]
        start += len(label[i])

    # 转换部首信息
    length_radical = []  # batch * （max_len-1）  不够的补零了
    for i in range(batch):
        length_tmp = []
        for j in range(max_length):
            if j < len(label[i]) - 1:
                length_tmp.append(len(char2radicals[label[i][j].lower()]) + 1)
            else:
                length_tmp.append(0)
        length_radical.append(length_tmp)
    length_radical = torch.Tensor(length_radical).long().cuda()

    max_radical_len = max(length_radical.view(-1))  # 找到最长的部首序列
    radical_input = torch.zeros(batch, max_length, max_radical_len).long().cuda()
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            # text_input[i][j + 1] = alp2num[label[i][j]]
            radicals = char2radicals[label[i][j].lower()]
            for k in range(len(radicals)):
                # radical_input[i][j][k+1] = alp2num_radical[radicals[k]] #串行执行
                radical_input[i][j][k + 1] = k + 1  # 并行执行

    sum_length = sum(length_radical.view(-1))
    radical_all = torch.zeros(sum_length).long().cuda()
    start = 0
    for i in range(batch):
        for j in range(len(label[i]) - 1):
            radicals = char2radicals[label[i][j].lower()]
            for k in range(len(radicals)):
                radical_all[start + k] = alp2num_radical[radicals[k]]
            radical_all[start + len(radicals)] = alp2num_radical['END']
            start += (len(radicals) + 1)

    # print("radical_input",radical_input)
    return length, text_input, text_all, length_radical, radical_input, radical_all, string_label