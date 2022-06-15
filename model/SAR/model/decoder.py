'''
This code is to construct decoder for SAR - two layer LSTMs combined with feature map with attention mechanism 
'''
import torch
import torch.nn as nn

__all__ = ['word_embedding','attention','decoder']

class word_embedding(nn.Module):
    def __init__(self, output_classes, embedding_dim):
        super(word_embedding, self).__init__()
        '''
        output_classes: number of output classes for the one hot encoding of a word
        embedding_dim: embedding dimension for a word
        '''
        self.linear = nn.Linear(output_classes, embedding_dim) # linear transformation

    def forward(self,x):
        x = self.linear(x)

        return x

class attention(nn.Module):
    def __init__(self, hidden_units, H, W, D):
        super(attention, self).__init__()
        '''
        hidden_units: hidden units of decoder
        H: height of feature map
        W: width of feature map
        D: depth of feature map
        '''
        self.conv1 = nn.Conv2d(hidden_units, D, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(D, D, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(D, 1, kernel_size=1, stride=1)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)
        self.H = H
        self.W = W
        self.D = D

    def forward(self, h, feature_map):
        '''
        h: hidden state from decoder output, with size [batch, hidden_units]
        feature_map: feature map from backbone network, with size [batch, channel, H, W]
        '''
        # reshape hidden state [batch, hidden_units] to [batch, hidden_units, 1, 1]
        h = h.unsqueeze(2)
        h = h.unsqueeze(3)
        h = self.conv1(h) # [batch, D, 1, 1]
        h = h.repeat(1, 1, self.H, self.W) # tiling to [batch, D, H, W]
        feature_map_origin = feature_map
        feature_map = self.conv2(feature_map) # [batch, D, H, W]
        combine = self.conv3(self.dropout(torch.tanh(feature_map + h))) # [batch, 1, H, W]
        combine_flat = combine.view(combine.size(0), -1) # resize to [batch, H*W]
        attention_weights = self.softmax(combine_flat) # [batch, H*W]
        attention_weights = attention_weights.view(combine.size()) # [batch, 1, H, W]
        glimpse = feature_map_origin * attention_weights.repeat(1, self.D, 1, 1) # [batch, D, H, W]
        glimpse = torch.sum(glimpse, dim=(2,3)) # [batch, D]

        return glimpse, attention_weights

class decoder(nn.Module):
    def __init__(self, output_classes, H, W, D=512, hidden_units=512, seq_len=40, device='cpu'):
        super(decoder, self).__init__()
        '''
        output_classes: number of output classes for the one hot encoding of a word
        H: feature map height
        W: feature map width
        D: glimpse depth
        hidden_units: hidden units of encoder/decoder for LSTM
        seq_len: output sequence length T
        '''
        self.linear1 = nn.Linear(output_classes, hidden_units)
        self.lstmcell = [nn.LSTMCell(hidden_units, hidden_units)] * 2
        # self.lstmcell1 = [nn.LSTMCell(hidden_units, hidden_units) for i in range(seq_len+1)]
        # self.lstmcell2 = [nn.LSTMCell(hidden_units, hidden_units) for i in range(seq_len+1)]
        self.attention = attention(hidden_units, H, W, D)
        self.linear2 = nn.Linear(hidden_units+D, output_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.seq_len = seq_len
        self.START_TOKEN = output_classes - 3 # Same as END TOKEN
        self.output_classes = output_classes
        self.hidden_units = hidden_units
        self.device = device

        self.lstmcell = torch.nn.ModuleList(self.lstmcell)
        # self.lstmcell1 = torch.nn.ModuleList(self.lstmcell1)
        # self.lstmcell2 = torch.nn.ModuleList(self.lstmcell2)

    def forward(self,hw,y,V):
        '''
        hw: embedded feature from encoder [batch, hidden_units]
        y: ground truth label one hot encoder [batch, seq, output_classes]
        V: feature map for backbone network [batch, D, H, W]
        '''
        outputs = []
        attention_weights = []
        batch_size = hw.shape[0]
        y_onehot = torch.zeros(batch_size, self.output_classes).to(self.device)
        for t in range(self.seq_len + 1):
            if t == 0:
                inputs_y = hw # size [batch, hidden_units]
                # LSTM layer 1 initialization:
                hx_1 = torch.zeros(batch_size, self.hidden_units).to(self.device) # initial h0_1
                cx_1 = torch.zeros(batch_size, self.hidden_units).to(self.device) # initial c0_1
                # LSTM layer 2 initialization:
                hx_2 = torch.zeros(batch_size, self.hidden_units).to(self.device) # initial h0_2
                cx_2 = torch.zeros(batch_size, self.hidden_units).to(self.device) # initial c0_2
            elif t == 1:
                y_onehot.zero_()
                y_onehot[:,self.START_TOKEN] = 1.0
                inputs_y = y_onehot
                inputs_y = self.linear1(inputs_y) # [batch, hidden_units]
            else:
                if self.training:
                    inputs_y = y[:,t-2,:] # [batch, output_classes]
                else:
                    # greedy search for now - beam search to be implemented!
                    index = torch.argmax(outputs[t-1], dim=-1) # [batch]
                    index = index.unsqueeze(1) # [batch, 1]
                    y_onehot.zero_()
                    inputs_y = y_onehot.scatter_(1, index, 1) # [batch, output_classes]

                inputs_y = self.linear1(inputs_y) # [batch, hidden_units_encoder]

            # LSTM cells combined with attention and fusion layer
            hx_1, cx_1 = self.lstmcell[0](inputs_y, (hx_1,cx_1))
            hx_2, cx_2 = self.lstmcell[1](hx_1, (hx_2,cx_2))
            glimpse, att_weights = self.attention(hx_2, V) # [batch, D], [batch, 1, H, W]
            combine = torch.cat((hx_2,glimpse), dim=1) # [batch, hidden_units_decoder+D]
            out = self.linear2(combine) # [batch, output_classes]
            out = self.softmax(out) # [batch, output_classes]
            outputs.append(out)
            attention_weights.append(att_weights)

        outputs = outputs[1:] # [seq_len, batch, output_classes]
        attention_weights = attention_weights[1:] # [seq_len, batch, 1, H, W]
        outputs = torch.stack(outputs) # [seq_len, batch, output_classes]
        outputs = outputs.permute(1,0,2) # [batch, seq_len, output_classes]
        attention_weights = torch.stack(attention_weights) # [seq_len, batch, 1, H, W]
        attention_weights = attention_weights.permute(1,0,2,3,4) # [batch, seq_len, 1, H, W]

        return outputs, attention_weights

# unit test
if __name__ == '__main__':

    batch_size = 2
    Height = 48
    Width = 160
    Channel = 512
    output_classes = 94
    embedding_dim = 512
    hidden_units = 512
    layers_decoder = 2
    seq_len = 40

    one_hot_embedding = torch.randn(batch_size, output_classes)
    one_hot_embedding[one_hot_embedding>0] = torch.ones(1)
    one_hot_embedding[one_hot_embedding<0] = torch.zeros(1)
    print("Word embedding size is:", one_hot_embedding.shape)

    embedding_model = word_embedding(output_classes, embedding_dim)
    embedding_transform = embedding_model(one_hot_embedding)
    print("Embedding transform size is:", embedding_transform.shape)

    hw = torch.randn(batch_size, hidden_units)
    feature_map = torch.randn(batch_size,Channel,Height,Width)
    print("Feature map size is:", feature_map.shape)

    attention_model = attention(hidden_units, Height, Width, Channel)
    glimpse, attention_weights = attention_model(hw, feature_map)
    print("Glimpse size is:", glimpse.shape)
    print("Attention weight size is:", attention_weights.shape)

    label = torch.randn(batch_size, seq_len, output_classes)
    decoder_model = decoder(output_classes, Height, Width, Channel, hidden_units, seq_len)
    outputs, attention_weights = decoder_model(hw, label, feature_map)
    print("Output size is:", outputs.shape)
    print("Attention_weights size is:", attention_weights.shape)
