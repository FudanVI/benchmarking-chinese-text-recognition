import torch
from torch.autograd import Variable
import collections
from torch.utils.data import ConcatDataset
import data.dataset as dataset

class strLabelConverter(object):
    def __init__(self, alphabet, ignore_case=False):
        self._ignore_case = ignore_case
        if self._ignore_case:
            alphabet = alphabet.lower()
        alphabet = list(alphabet)
        alphabet.append('BLANK')
        self.alphabet = alphabet
        self.dict = {}
        for i, char in enumerate(alphabet):
            self.dict[char] = i + 1

    def encode(self,text):
        if isinstance(text, str):
            text_raw = text
            text = []
            for char in text_raw:
                if char not in self.dict.keys():
                    text.append(0)
                else:
                    text.append(self.dict[char])

            length = len(text)
        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.IntTensor(text), torch.IntTensor(length))

    def decode(self, texts, lengths, raw = False):
        if lengths.numel() == 1:
            length = lengths[0]
            if raw:
                return ''.join([self.alphabet[i-1] for i in texts])
            else:
                text = []
                for i in range(length):
                    if (texts[i] != 0) and (not(i > 0 and texts[i] == texts[i-1])):
                        text.append(self.alphabet[texts[i] - 1])
                return ''.join(text)
        else:
            res = []
            index = 0
            for i in range(lengths.numel()):
                res.append(self.decode(texts[index : index + lengths[i]], torch.IntTensor([lengths[i]]), raw = raw))
                index += lengths[i]
            return res



class averager(object):
    def __init__(self):
        self.reset()

    def add(self, cost):
        if isinstance(cost, Variable):
            self.count = cost.data.numel()
            cost = cost.data.sum()
        elif isinstance(cost, torch.Tensor):
            self.count = cost.numel()
            cost = cost.sum()
        self.n_count += self.count
        self.sum += cost

    def reset(self):
        self.n_count = 0
        self.sum = 0
        self.count = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def Q2B(uchar):
    inside_code = ord(uchar)
    if inside_code == 0x3000:
        inside_code = 0x0020
    else:
        inside_code -= 0xfee0
    if inside_code < 0x0020 or inside_code > 0x7e:
        return uchar
    return chr(inside_code)

def stringQ2B(ustring):
    return "".join([Q2B(uchar) for uchar in ustring])

def get_data(args):
    test_dataset = dataset.lmdbDataset(root=args.test_dataset)
    if not args.test:
        train_list = args.train_dataset.split(',')
        train_dataset_list = [dataset.lmdbDataset(root=path) for path in train_list]
        train_dataset = ConcatDataset(train_dataset_list)
        return train_dataset, test_dataset
    else:
        return None, test_dataset