import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class Embedding(nn.Module):
    def __init__(self, in_timestep, in_planes, mid_dim = 4096, embed_dim=300):
        super(Embedding, self).__init__()
        self.in_timestep = in_timestep
        self.in_planes = in_planes
        self.embed_dim = embed_dim
        self.mid_dim = mid_dim
        self.avgpool = nn.AdaptiveMaxPool2d((1,32))
        self.eEmbed = nn.Linear(in_timestep * in_planes, self.embed_dim)  # Embed encoder output to a word-embedding like

    def forward(self, x):
        # print(self.in_timestep,self.in_planes,self.embed_dim)
        B,C,D = x.shape
        x = self.avgpool(x.view(B,C,8,64)).contiguous().view(B,C,-1)
        # print("x",x.shape,self.in_planes)
        x = x.reshape((x.shape[0], -1))
        # print("x.shape", x.shape)
        x = self.eEmbed(x)

        return x

# Log 10.30 self-attention based
class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn

class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention Module"""
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)

        nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5))
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)
        nn.init.xavier_normal_(self.fc.weight)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, _ = q.size()
        sz_b, len_k, _ = k.size()
        sz_b, len_v, _ = v.size()

        residual = q

        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        q = q.permute(2, 0, 1, 3).contiguous().view(-1, len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # (n*b) x lv x dv

        output, attn = self.attention(q, k, v)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Conv1d(d_in, d_hid, 1) # position-wise
        self.w_2 = nn.Conv1d(d_hid, d_in, 1) # position-wise
        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        output = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)
        return output

class self_block(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(self_block, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        # enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        # enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class Embedding_self_att(nn.Module):
    def __init__(self, in_timestep, in_planes, n_head, n_layers, embed_dim=300):
        super(Embedding_self_att, self).__init__()
        self.in_timestep = in_timestep
        self.in_planes = in_planes
        self.n_head = n_head
        self.n_layers = n_layers
        self.embed_dim = embed_dim
        # self.attention = MultiHeadAttention(n_head=self.n_head,
        #                                     d_model=self.in_planes,
        #                                     d_k=self.in_planes,
        #                                     d_v=self.in_planes)
        # self.ffn = PositionwiseFeedForward(d_in=self.in_planes, d_hid=self.in_planes)
        # self.attention_block = self_block(d_model=self.in_planes,
        #                                   d_inner=self.in_planes,
        #                                   n_head=self.n_head,
        #                                   d_k=self.in_planes,
        #                                   d_v=self.in_planes)
        self. proj = nn.Linear(self.in_timestep * self.in_planes, self.embed_dim)
        self.layer_stack = nn.ModuleList([self_block(d_model=self.in_planes,
                                          d_inner=self.in_planes,
                                          n_head=self.n_head,
                                          d_k=self.in_planes,
                                          d_v=self.in_planes) for _ in range(self.n_layers)])

    def forward(self, x):
        N = x.size(0)
        # x, att = self.attention(x, x, x)
        # x = self.ffn(x)

        for enc_layer in self.layer_stack:
            x, enc_slf_attn = enc_layer(x)

        x = x.reshape((N, -1))
        x = self.proj(x)

        return x