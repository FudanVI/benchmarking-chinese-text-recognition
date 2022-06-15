import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math, copy
import numpy as np
import time
from torch.autograd import Variable

class RadicalBranch(nn.Module):

    def __init__(self, radical_alphabet_len):
        super(RadicalBranch, self).__init__()

        # 从config文件里读
        self.radical_n_class = radical_alphabet_len
        self.attention_compress = nn.Linear(4, 1)
        self.features_compress = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=1, stride=1, padding=0)
        self.embedding_radical = Embeddings(256, self.radical_n_class)
        self.pe_radical = PositionalEncoding(d_model=256, dropout=0.1, max_len=8000)
        self.decoder_radical = Decoder()
        self.generator_radical = Generator(512, self.radical_n_class)

    def forward(self, attention_map, conv_feature, radical_input, radical_length):
        # attention_map : batch * 4 * len * (H * W)
        # conv_feature : batch * channel * H * W
        # radical_input : batch * l * rl  rl表示样本中最长的部首长度
        # radical_length : batch * l   表示每个样本中每个字符的部首长度
        # conv_feature 高度等于8
        attention_map_tmp = attention_map.permute(0, 2, 3, 1)
        attention_map_tmp = attention_map_tmp.squeeze(3)  # batch * len * (H*W)
        b, c, h, w = conv_feature.size()  # H W => 8 * 64
        conv_feature_tmp = conv_feature.view(b, c, -1)
        
        char_maps = torch.mul(attention_map_tmp.unsqueeze(2), conv_feature_tmp.unsqueeze(1))
        char_maps = self.features_compress(char_maps.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)  # batch * len * channel * (H*W)
        b, l, c, hw = char_maps.size()
        char_maps = char_maps.contiguous().view(b * l, c, 8, 8)

        b, l, rl = radical_input.size()
        embedding_radical = self.embedding_radical(radical_input.view(-1, rl))
        postion_embedding_radical = self.pe_radical(torch.zeros(embedding_radical.shape).cuda()).cuda()
        radical_input_with_pe = torch.cat([embedding_radical, postion_embedding_radical], 2)

        radical_input_with_pe, attention_map = self.decoder_radical(radical_input_with_pe, char_maps)
        radical_decoder_result = self.generator_radical(radical_input_with_pe)

        # print('radical_decoder_result size:', radical_decoder_result.size())

        total_radical_length = torch.sum(radical_length.view(-1)).data
        # print('total_radical_length:', total_radical_length)
        probs_res_radical = torch.zeros(total_radical_length, self.radical_n_class).type_as(radical_decoder_result.data)
        start = 0
        for index, length in enumerate(radical_length.view(-1)):
            # print("index, length", index,length)
            length = length.data
            if length == 0:
                continue
            probs_res_radical[start:start + length, :] = radical_decoder_result[index, 0:0 + length, :]
            start = start + length

        return {
            'radical_pred': probs_res_radical,
            'map': attention_map,
            'conv': conv_feature,
        }

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=8000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        '''这个地方去掉，做pe的对照实验'''
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        self.relu = nn.ReLU()

    def forward(self, x):
        # return F.softmax(self.proj(x))
        return self.proj(x)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.d_model)
        # print("embed",embed)
        # embed = self.lut(x)
        # print(embed.requires_grad)
        return embed

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.mask_multihead = MultiHeadedAttention(h=4, d_model=512, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=512)

        '''
        工程：这里把head修改为1
        '''
        self.multihead = MultiHeadedAttention(h=4, d_model=512, dropout=0.1, compress_attention=False)
        self.mul_layernorm2 = LayerNorm(features=512)

        self.pff = PositionwiseFeedForward(512, 1024)
        self.mul_layernorm3 = LayerNorm(features=512)

    def forward(self, text, conv_feature):
        text_max_length = text.shape[1]
        mask = subsequent_mask(text_max_length).cuda()

        result = text
        result = self.mul_layernorm1(result + self.mask_multihead(result, result, result, mask=mask)[0])

        b, c, h, w = conv_feature.shape
        conv_feature = conv_feature.view(b, c, h * w).permute(0, 2, 1).contiguous()

        # print('result size:', result.size())
        # print('conv_feature size:', conv_feature.size())
        word_image_align, attention_map = self.multihead(result, conv_feature, conv_feature, mask=None)
        result = self.mul_layernorm2(result + word_image_align)

        result = self.mul_layernorm3(result + self.pff(result))

        return result, attention_map

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # cnt = 0
        # for l , x in zip(self.linears, (query, key, value)):
        #     print(cnt,l,x)
        #     cnt += 1

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # print("在Multi中，query的尺寸为", query.shape)
        # print("在Multi中，key的尺寸为", key.shape)
        # print("在Multi中，value的尺寸为", value.shape)

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        # 因为multi-head可能会产生若干张注意力map，应该将其合并为一张
        if self.compress_attention:
            batch, head, s1, s2 = attention_map.shape
            attention_map = attention_map.permute(0, 2, 3, 1).contiguous()
            attention_map = self.compress_attention_linear(attention_map).permute(0, 3, 1, 2).contiguous()
        # print('transformer file:', attention_map)

        return self.linears[-1](x), attention_map


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    '''
    这里使用偏移k=2是因为前面补位embedding
    '''
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None, align=None):
    "Compute 'Scaled Dot Product Attention'"

    # print(mask)
    # print("在attention模块,q_{0}".format(query.shape))
    # print("在attention模块,k_{0}".format(key.shape))
    # print("在attention模块,v_{0}".format(key.shape))
    # print("mask :",mask)
    # print("mask的尺寸为",mask.shape)

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # print(mask)
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass
        # print("scores", scores.shape)
    '''
    工程
    这里的scores需要再乘上一个prob
    这个prob的序号要和word_index对应！
    '''

    # if align is not None:
    #     # print("score", scores.shape)
    #     # print("align", align.shape)
    #
    #     scores = scores * align.unsqueeze(1)

    p_attn = F.softmax(scores, dim=-1)

    # if mask is not None:
    #     print("p_attn",p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print("p_attn", p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


if __name__ == '__main__':
    att_f = torch.randn((16, 4, 22, 512))
    conv_f = torch.randn((16, 1024, 8, 64))
    radical_input = torch.ones((16, 22, 10)).long()
    radical_length = torch.ones((16, 22))
    model = RadicalBranch(36, ).cuda()
    print("radical_input", radical_input.dtype)
    output = model(att_f.cuda(), conv_f.cuda(), radical_input.cuda(), radical_length)
    print(output.shape)

    # attention_map
    # size: torch.Size([16, 4, 22, 512])
    # conv_feature
    # size: torch.Size([16, 1024, 8, 64])
    # radical_input
    # size: torch.Size([16, 22, 10])
    # radical_length
    # size: torch.Size([16, 22])
