'''
This code is to construct encoder for SAR - two layer LSTMs
'''
import torch
import torch.nn as nn

__all__ = ['encoder']

class encoder(nn.Module):
    def __init__(self, H, C, hidden_units=512, layers=2, keep_prob=1.0, device='cpu'):
        super(encoder, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=(H,1), stride=1)
        self.lstm = nn.LSTM(input_size=C, hidden_size=hidden_units, num_layers=layers, batch_first=True, dropout=keep_prob)
        self.layers = layers
        self.hidden_units = hidden_units
        self.device = device

    def forward(self, x):
        self.lstm.flatten_parameters()
        # x is feature map in [batch, C, H, W]
        # Initialize hidden state with zeros
        h_0 = torch.zeros(self.layers*1, x.size(0), self.hidden_units).to(self.device)
        # Initialize cell state
        c_0 = torch.zeros(self.layers*1, x.size(0), self.hidden_units).to(self.device)
        x = self.maxpool(x) # [batch, C, 1, W]
        x = torch.squeeze(x) # [batch, C, W]
        if len(x.size()) == 2: # [C, W]
            x = x.unsqueeze(0) # [batch, C, W]
        x = x.permute(0,2,1) # [batch, W, C]
        _, (h, _) = self.lstm(x, (h_0, c_0)) # h with shape [layers*1, batch, hidden_uints]

        return h[-1] # shape [batch, hidden_units]

# unit test
if __name__ == '__main__':

    batch_size = 32
    Height = 48
    Width = 160
    Channel = 3
    input_feature = torch.randn(batch_size,Channel,Height,Width)
    print("Input feature size is:",input_feature.shape)

    encoder_model = encoder(Height, Channel, hidden_units=512, layers=2, keep_prob=1.0)
    output_encoder = encoder_model(input_feature)

    print("Output feature of encoder size is:",output_encoder.shape) # (batch, hidden_units)