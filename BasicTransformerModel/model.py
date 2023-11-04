import math
import os
from tempfile import TemporaryDirectory
from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import dataset

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, v_size: int):
        super().__init__()
        self.d_model = d_model
        self.v_size = v_size
        # Create the embedding layer
        self.embedding = nn.Embedding(v_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float, device):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(dropout)

        # Matrix of PE
        self.encoding = torch.zeros(max_len, d_model, device = device)
        self.encoding.requires_grad = False
        # Matrix of Position
        pos = torch.arange(0,max_len,device=device)
        pos = pos.float().unsqueeze(dim=1)

        _2i = torch.arange(0, max_len, step = 2, device = device)
        self.encoding[:, 0::2] = torch.sin(pos / (10000.0 ** (_2i / d_model)))
        self.encoding[:, 0::1] = torch.cos(pos / (10000.0 ** (_2i / d_model)))

        self.encoding.unsqueeze(0)
        self.register_buffer('pe',self.encoding)
    
    def forward(self, x):
        batch_size, seq_len = x.size()
        x = x + (self.encoding[:, :seq_len, :])
        return self.dropout(x)

class AddNorm(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(AddNorm, self).__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(size, eps=eps)
    
    def forward(self, x, sublayer):       
        return self.norm(x)

class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout) -> None:
        super().__init__()    
        self.dropout = nn.Dropout(dropout)
        
        self.linear_1 = nn.Linear(d_model,d_ff) #W1 + Bias1
        self.linear_2 = nn.Linear(d_ff,d_model) #W2 + Bias2

    def forward(self, x):
        #FNN(x) = ReLU(0, xW1 + b1)W2 + b2
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttetion(nn.Module):

    def __init__(self, d_model: int, head: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.head = head
        assert d_model % head == 0 # d_model has to be divisable by head

        self.d_k = d_model // head
        self.w_q = nn.Linear(d_model,d_model) # Wq Weight for Query
        self.w_k = nn.Linear(d_model,d_model) # Wk Weight for Key
        self.w_v = nn.Linear(d_model,d_model) # Wv Weight for Value
        self.w_o = nn.Linear(d_model,d_model) # Wo Weight for Output

        self.dropout = nn.Dropout(dropout)

    def forward(self,q,k,v,mask):

        batch_size = q.size(0)
        # Apply the linear transformations and split into `head` heads
        query = self.w_q(q).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        key = self.w_k(k).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
        value = self.w_v(v).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)

        # Attention(Q,K,V) = softmax(QK^T/d_k^0.5) V
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:                 # Apply mask to multi-heads
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        
        attention_scores = attention_scores.softmax(dim=-1)
        attention_scores = self.dropout(attention_scores)

        # Multihead(Q,K,V) = Concat(head1...head_n) Wo        
        multihead = (attention_scores @ value).transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        multihead = self.w_o(multihead)

        return multihead

