import torch
import torch.nn as nn
import math
from .utils.transformer_utils import *
from .embedding.transformer_embedding import *
from .attention.transformer_attention import *
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask=None):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        #return self.norm(x).view(x.size(0),-1)
        return self.norm(x)

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask=None):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

def make_model(N,d_model,h,d_ff,seq_len,vocab_size,dropout=0.1):
    '''
    N: number of stack
    d_model: d_model
    h: head
    d_ff: inner hidden layer
    input_size: this is for final DNN
    output_size: this is for final DNN
    '''
    c = copy.deepcopy
    attn = MultiHeadedAttention(h,d_model)
    FFN = PositionwiseFeedForward(d_model,d_ff)
    enc = EncoderLayer(d_model,c(attn),c(FFN),dropout)
    final_encoder = Encoder(enc,N)
    word_embedding = Embeddings(d_model,vocab_size)
    pos_emb = PositionalEncoding(d_model,dropout)
    
    final_model = nn.Sequential(
        final_encoder
    )
    
    for p in final_model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return final_model,word_embedding,pos_emb