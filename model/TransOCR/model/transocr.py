import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy
import numpy as np
from torch.autograd import Variable

from utils import get_alphabet

class Bottleneck(nn.Module):

    def __init__(self, input_dim):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, input_dim, 1)
        self.bn1 = nn.BatchNorm2d(input_dim)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(input_dim, input_dim, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(input_dim)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    '''
    构成ResNet的残差块
    '''

    def __init__(self, inplanes, planes, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample != None:
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, num_in, block, layers, args):
        super(ResNet, self).__init__()

        self.args = args

        self.conv1 = nn.Conv2d(num_in, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d((2, 2), (2, 2))

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU(inplace=True)

        self.layer1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer1 = self._make_layer(block, 128, 256, layers[0])
        self.layer1_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer1_bn = nn.BatchNorm2d(256)
        self.layer1_relu = nn.ReLU(inplace=True)

        self.layer2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer2 = self._make_layer(block, 256, 256, layers[1])
        self.layer2_conv = nn.Conv2d(256, 256, 3, 1, 1)
        self.layer2_bn = nn.BatchNorm2d(256)
        self.layer2_relu = nn.ReLU(inplace=True)

        self.layer3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer3 = self._make_layer(block, 256, 512, layers[2])
        self.layer3_conv = nn.Conv2d(512, 512, 3, 1, 1)
        self.layer3_bn = nn.BatchNorm2d(512)
        self.layer3_relu = nn.ReLU(inplace=True)

        self.layer4_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.layer4 = self._make_layer(block, 512, 512, layers[3])
        self.layer4_conv2 = nn.Conv2d(512, 1024, 3, 1, 1)
        self.layer4_conv2_bn = nn.BatchNorm2d(1024)
        self.layer4_conv2_relu = nn.ReLU(inplace=True)

    def _make_layer(self, block, inplanes, planes, blocks):

        if inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, 3, 1, 1),
                nn.BatchNorm2d(planes), )
        else:
            downsample = None
        layers = []
        layers.append(block(inplanes, planes, downsample))
        # self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, planes, downsample=None))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.layer1_pool(x)
        x = self.layer1(x)
        x = self.layer1_conv(x)
        x = self.layer1_bn(x)
        x = self.layer1_relu(x)

        if not self.args.radical:
            x = self.layer2_pool(x)
        x = self.layer2(x)
        x = self.layer2_conv(x)
        x = self.layer2_bn(x)
        x = self.layer2_relu(x)

        if not self.args.radical:
            x = self.layer3_pool(x)
        x = self.layer3(x)
        x = self.layer3_conv(x)
        x = self.layer3_bn(x)
        x = self.layer3_relu(x)

        # x = self.layer4_pool(x)
        x = self.layer4(x)
        x = self.layer4_conv2(x)
        x = self.layer4_conv2_bn(x)
        x = self.layer4_conv2_relu(x)

        return x


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len=7000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if self.compress_attention:
            batch, head, s1, s2 = attention_map.shape
            attention_map = attention_map.permute(0, 2, 3, 1).contiguous()
            attention_map = self.compress_attention_linear(attention_map).permute(0, 3, 1, 2).contiguous()
        return self.linears[-1](x), attention_map


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None, align=None):

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass
    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.proj(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        return embed

class TransformerBlock(nn.Module):
    """Transformer Block"""
    def __init__(self, dim, num_heads, ff_dim, dropout):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadedAttention(h=num_heads, d_model=dim, dropout=dropout)
        self.proj = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.pwff = PositionwiseFeedForward(dim, ff_dim)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(x)
        h = self.drop(self.proj(self.attn(x, x, x, mask)[0]))
        x = x + h
        h = self.drop(self.pwff(self.norm2(x)))
        x = x + h
        return x

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=1024)

        self.multihead = MultiHeadedAttention(h=4, d_model=1024, dropout=0.1, compress_attention=False)
        self.mul_layernorm2 = LayerNorm(features=1024)

        self.pff = PositionwiseFeedForward(1024, 2048)
        self.mul_layernorm3 = LayerNorm(features=1024)

    def forward(self, text, conv_feature):
        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length).cuda()

        result = text
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])

        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()
        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None)
        result = self.mul_layernorm2(result + word_image_align)

        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map


class Transformer(nn.Module):

    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        alphabet = get_alphabet(args, 'char')
        self.word_n_class = len(alphabet)
        self.embedding_word = Embeddings(512, self.word_n_class)
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=7000)
        self.encoder = ResNet(num_in=3, block=BasicBlock, layers=[3,4,6,3], args=args).cuda()
        self.decoder = Decoder()
        self.generator_word = Generator(1024, self.word_n_class)

        if args.radical:
            radical_alphabet = get_alphabet(args, 'radical')
            self.radical_n_class = len(radical_alphabet)
            self.attention_compress = nn.Linear(4, 1)
            self.features_compress = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=1, stride=1, padding=0)
            self.embedding_radical = Embeddings(512, self.radical_n_class)
            self.pe_radical = PositionalEncoding(d_model=512, dropout=0.1, max_len=7000)
            self.decoder_radical = Decoder()
            self.generator_radical = Generator(1024, self.radical_n_class)

            self.sub_channel = nn.Linear(2048, 1024)

            self.vit_radical = TransformerBlock(1024, 4, 2048, dropout=0.1)
            self.class_token = nn.Parameter(torch.zeros(1, 1, 1024))
            self.pe_radical_overall = PositionalEncoding(d_model=1024, dropout=0.1, max_len=7000)
            self.sub_channel_1 = nn.Linear(2048, 1024)
            self.pre_logits = nn.Linear(1024, 1024)
            self.norm = nn.LayerNorm(1024, eps=1e-6)
            self.fc = nn.Linear(1024, self.word_n_class)
            self.generator_word_overall = Generator(1024, self.word_n_class)


    def forward(self, image, text_length, text_input, radical_length=None, radical_input=None, conv_feature=None, test=False, att_map=None):

        if conv_feature is None:
            conv_feature = self.encoder(image)

        if text_length is None:
            return {
                'conv': conv_feature,
            }

        text_embedding = self.embedding_word(text_input)
        postion_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda()
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2)
        text_input_with_pe, attention_map = self.decoder(text_input_with_pe, conv_feature)
        word_decoder_result = self.generator_word(text_input_with_pe)

        if test == False and self.args.radical:
            attention_map_tmp = attention_map.permute(0, 2, 3, 1)
            attention_map_tmp = self.attention_compress(attention_map_tmp).squeeze(3)  # batch * len * (H*W)
            b, c, h, w = conv_feature.size()  # H W => 8 * 64
            conv_feature_tmp = conv_feature.view(b, c, -1)

            char_maps = torch.mul(attention_map_tmp.unsqueeze(2), conv_feature_tmp.unsqueeze(1))
            char_maps = self.features_compress(char_maps.permute(0, 3, 1, 2)).permute(0, 2, 3,
                                                                                      1)  # batch * len * channel * (H*W)
            b, l, c, hw = char_maps.size()
            char_maps = char_maps.contiguous().view(b * l, c, 8, 8)

            b, l, rl = radical_input.size()
            embedding_radical = self.embedding_radical(radical_input.view(-1, rl))
            postion_embedding_radical = self.pe_radical(torch.zeros(embedding_radical.shape).cuda()).cuda()
            radical_input_with_pe = torch.cat([embedding_radical, postion_embedding_radical], 2)
            radical_input_with_pe, attention_map = self.decoder_radical(radical_input_with_pe, char_maps)
            radical_decoder_result = self.generator_radical(radical_input_with_pe)

        if test:
            return {
                'pred': word_decoder_result,
                'map': attention_map,
                'conv': conv_feature,
            }

        else:
            total_length = torch.sum(text_length).data
            probs_res = torch.zeros(total_length, self.word_n_class).type_as(word_decoder_result.data)
            start = 0
            for index, length in enumerate(text_length):
                length = length.data
                probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
                start = start + length

            probs_res_radical = None
            if self.args.radical:
                total_radical_length = torch.sum(radical_length.view(-1)).data
                probs_res_radical = torch.zeros(total_radical_length, self.radical_n_class).type_as(
                    radical_decoder_result.data)
                start = 0
                for index, length in enumerate(radical_length.view(-1)):
                    length = length.data
                    if length == 0:
                        continue
                    probs_res_radical[start:start + length, :] = radical_decoder_result[index, 0:0 + length, :]
                    start = start + length

            return {
                'pred': probs_res,
                'radical_pred': probs_res_radical,
                'map': attention_map,
                'conv': conv_feature,
            }

if __name__ == '__main__':
    net = ResNet(num_in=3, block=BasicBlock, layers=[3, 4, 6, 3]).cuda()
    image = torch.Tensor(8, 3, 64, 64).cuda()
    result = net(image)
    print(result.shape)
    pass


