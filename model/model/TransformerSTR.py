import torch
import torch.nn as nn

from model.ResNet import ResNet, BasicBlock
from util import get_alphabet
from model.TransformerUtil import Decoder, PositionalEncoding, Embeddings, Generator

alphabet = get_alphabet()

class Transformer(nn.Module):

    def __init__(self):
        super(Transformer, self).__init__()

        self.word_n_class = len(alphabet)
        self.pe = PositionalEncoding(d_model=512, dropout=0.1, max_len=7000)
        self.encoder = ResNet(num_in=3, block=BasicBlock, layers=[3,4,6,3]).cuda()

        self.decoder = Decoder()

        for p in self.parameters():
            p.requires_grad = False

        self.embedding_word = Embeddings(512, self.word_n_class)
        self.generator_word = Generator(1024, self.word_n_class)


    def forward(self, image, text_length, text_input, conv_feature=None, test=False, att_map=None):
        if conv_feature is None:
            conv_feature = self.encoder(image)

        if text_length is None:
            return {
                'conv': conv_feature,
            }

        text_embedding = self.embedding_word(text_input)
        postion_embedding = self.pe(torch.zeros(text_embedding.shape).cuda()).cuda()
        text_input_with_pe = torch.cat([text_embedding, postion_embedding], 2)
        batch, seq_len, _ = text_input_with_pe.shape

        text_input_with_pe, attention_map = self.decoder(text_input_with_pe, conv_feature)
        word_decoder_result = self.generator_word(text_input_with_pe)

        if test:
            #-----testing phase----
            return {
                'pred': word_decoder_result,
                'map': attention_map,
                'conv': conv_feature,
            }

        else:
            #-----traing phase-----
            total_length = torch.sum(text_length).data
            probs_res = torch.zeros(total_length, self.word_n_class).type_as(word_decoder_result.data)

            start = 0
            for index, length in enumerate(text_length):
                length = length.data
                probs_res[start:start + length, :] = word_decoder_result[index, 0:0 + length, :]
                start = start + length

            return {
                'pred': probs_res,
                'map': attention_map,
                'conv': conv_feature,
            }


