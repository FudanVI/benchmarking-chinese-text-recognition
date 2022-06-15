import torch.nn as nn
import torch

class BiLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BiLSTM, self).__init__()
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden*2, nOut)

    def forward(self, input):
        if not hasattr(self, '_flattened'):
            self.rnn.flatten_parameters()
            setattr(self, '_flattened', True)

        rnnOut, _ = self.rnn(input)
        T, b, c = rnnOut.size()
        rnnOut = rnnOut.view(T*b, c)

        output = self.embedding(rnnOut)
        output = output.view(T, b, -1)

        return output


class CRNN(nn.Module):
    def __init__(self, nc, nh, nclass, height, LeakyRelu=False):
        super(CRNN, self).__init__()

        kernal_size = [3, 3, 3, 3, 3, 3, 3]
        padding_size = [1, 1, 1, 1, 1, 1, 1]
        stride_size = [1, 1, 1, 1, 1, 1, 1]
        channels = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def convRelu(i, BatchNormalize=False):
            if i == 0:
                nIn = nc
            else:
                nIn = channels[i-1]
            nOut = channels[i]
            cnn.add_module('conv{0}'.format(i),
                           nn.Conv2d(nIn, nOut, kernal_size[i], stride_size[i], padding_size[i]))
            if BatchNormalize:
                cnn.add_module('batchnorm{0}'.format(i), nn.BatchNorm2d(nOut))
            if LeakyRelu:
                cnn.add_module('relu{0}'.format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module('relu{0}'.format(i), nn.ReLU(True))
        convRelu(0)
        cnn.add_module('pooling{0}'.format(0), nn.MaxPool2d((2, 2), (1, 2), (1, 0)))
        convRelu(1)
        cnn.add_module('pooling{0}'.format(1), nn.MaxPool2d((2, 2), (1, 2), (1, 0)))
        convRelu(2, True)
        convRelu(3)
        cnn.add_module('pooling{0}'.format(2),
                       nn.MaxPool2d((2,2), (2,1), (0,1)))
        convRelu(4, True)
        convRelu(5)
        cnn.add_module('pooling{0}'.format(3),
                       nn.MaxPool2d((2,2), (2,1), (0,1)))
        convRelu(6, True)

        self.cnn = cnn

        self.avg_pooling = nn.AvgPool2d(kernel_size=(height//4, 1), stride=(height//4, 1))

        self.rnn = nn.Sequential(
            BiLSTM(512, nh, nh),
            BiLSTM(nh, nh, nclass)
        )


    def forward(self, input):
        conv = self.cnn(input)
        conv = self.avg_pooling(conv)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output

if __name__=="__main__":
    img = torch.randn(60, 3, 64, 100).cuda(1)
    crnn = CRNN(3,256, 36, 64).cuda(1)
    res = crnn(img)
    print(res.size())
