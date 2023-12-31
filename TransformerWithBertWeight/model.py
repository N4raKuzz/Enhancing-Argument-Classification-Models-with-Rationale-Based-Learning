import math
import torch
import numpy as np
from torch import nn, Tensor
from transformers import BertModel

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, v_size: int):
        super().__init__()
        self.d_model = d_model
        self.v_size = v_size
        self.embedding = nn.Embedding(v_size,d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int, dropout: float, device):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Matrix of PE
        self.encoding = torch.zeros(max_len, d_model).to(device) 
        self.encoding.requires_grad = False
        # Matrix of Position
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device) 

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)).to(device) 
        self.encoding[:, 0::2] = torch.sin(pos * div_term)
        self.encoding[:, 1::2] = torch.cos(pos * div_term)

        self.encoding = self.encoding.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',self.encoding)
    
    def forward(self, x):
        x = x + (self.encoding[:x.size(0), :])
        return self.dropout(x)

class Normalization(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(Normalization, self).__init__()
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

    def __init__(self, d_model: int, head: int, bert_attention, dropout = 0.1) -> None:
        super().__init__()
        self.d_model = d_model
        self.head = head
        assert d_model % head == 0 # d_model has to be divisable by head

        self.d_k = d_model // head
        self.w_q = bert_attention.self.query # Wq Weight for Query
        self.w_k = bert_attention.self.key # Wk Weight for Key
        self.w_v = bert_attention.self.value # Wv Weight for Value
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

class ResidualConnection(nn.Module):

    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = Normalization(size)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x))) # Add & Norm

class Encoder(nn.Module):

    def __init__(self, size : int, multi_attention_layer, feed_forward, dropout = 0.1) -> None:
        super().__init__()
        self.multi_attention_layer = multi_attention_layer
        self.feed_forward = feed_forward
        self.ma_rc = ResidualConnection(size, dropout)  # The first residual connection to do for the multi-attention layer
        self.ff_rc = ResidualConnection(size, dropout)  # The second residual connection to do for the feed-forward layer

    def forward(self, x):
        x = self.ma_rc(x,self.multi_attention_layer(x,x,x,None))
        x = self.ff_rc(x,self.feed_forward(x))
        return x

class LinearLayer(nn.Module):

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Linear(d_model,1)
    
    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim = -1)

class TransformerBertWeight(nn.Module):

    def __init__(self, num_classes:int, d_model:int, d_ff:int, input_embedding : InputEmbedding, positional_encoding : PositionalEncoding, device, num_heads : int = 8, dropout : float = 0.1) -> None:
        super().__init__()

        self.device = device
        self.to(device)

        bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
        self.encoders = []
        self.embed = input_embedding
        self.pos_enc = positional_encoding

        for i in range(11):
            # Retrieve weights from encoder layer of BERT
            bert_attention = bert_model.encoder.layer[i].attention
            
            mha = MultiHeadAttetion(d_model, num_heads, bert_attention, dropout).to(device) # Create a MultiHeadAttention layer using BERT weights
            ff = FeedForward(d_model, d_ff, dropout).to(device) # Create FeedForward layer
            encoder = Encoder(d_model, mha, ff, dropout).to(device) # Create Encoder
            encoder.requires_grad = False  # Freeze layer
            self.encoders.append(encoder)
                                                                                                                                                                                                                                                                
        bert_attention = bert_model.encoder.layer[11].attention
        mha = MultiHeadAttetion(d_model, num_heads, bert_attention, dropout).to(device) # Create a MultiHeadAttention layer using BERT weights
        ff = FeedForward(d_model, d_ff, dropout).to(device) # Create FeedForward layer
        encoder = Encoder(d_model, mha, ff, dropout).to(device) # Create the 12th layer that trained on
        self.encoders.append(encoder)

        #self.dense1 = nn.Linear(d_model, num_classes).to(device)
        self.dense = LinearLayer(d_model).to(device)
    
    def forward(self, src):
        src = self.embed(src)
        src = self.pos_enc(src)
        for encoder in self.encoders:
            src = encoder(src)

        src = self.dense(src[:, -1, :])
        return src

    def loss_cross_entropy_softmax(self, x, truth):
   
        x_shift = x - np.max(x) # Remove the Xmax to avoid overflow [m,1]
        x_exp = np.exp(x_shift)
        y_tilde = x_exp / np.sum(x_exp) 
        y_log = np.log(y_tilde+1e-9) 

        l = -np.sum(np.multiply(y_log, truth))  # Loss
        dl_dy_tilde = -np.multiply(1/(y_tilde+1e-9), truth) 
        dy_dx = -np.dot(y_tilde, y_tilde.T) + np.diag(y_tilde.flatten()) 
        dl_dx = np.dot(dl_dy_tilde.T, dy_dx) #The loss derivative with respect to x

        return l, dl_dx

    def linear(self, x):
        return self.linear_layer(x)
    
    
# Do not need a decoder block for text classification: Only for model to generate text
# class Decoder(nn.Module):

#     def __init__(self, self_attention, cross_attention, feed_forward, dropout: float) -> None:
#         super().__init__()
#         self.self_attention = self_attention
#         self.cross_attention = cross_attention
#         self.feed_forward = feed_forward
#         self.sa_rc = ResidualConnection(dropout)  # The first residual connection to do for the self-attention layer
#         self.ca_rc = ResidualConnection(dropout) # The second residual connection to do for the cross-attention layer
#         self.ff_rc = ResidualConnection(dropout)  # The third residual connection to do for the feed-forward layer
    

#     def forward(self, x, encoder_output, src_mask, tgt_mask):
#         x = self.sa_rc(x,lambda x: self.multi_attention_layer(x,x,x,src_mask))
#         x = self.ca_rc(x,lambda x:self.multi_attention_layer(x,encoder_output,encoder_output,tgt_mask))
#         x = self.ff_rc(x,self.feed_forward(x))
#         return x
    

# class Transformer(nn.Module):

#     def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbedding, tgt_embed: InputEmbedding, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, linear_layer: LinearLayer) -> None:
#         super().__init__()
#         self.encoder = encoder
#         self.decoder = decoder
#         self.src_embed = src_embed
#         self.tgt_embed = tgt_embed
#         self.src_pos = src_pos
#         self.tgt_pos = tgt_pos
#         self.linear_layer = linear_layer

#     def encode(self, src, src_mask):
#         src = self.src_embed(src)
#         src = self.src_pos(src)
#         return self.encoder(src, src_mask)
    
#     def decode(self, encoder_output: torch.Tensor, src_mask, tgt, tgt_mask):
#         tgt = self.tgt_embed(tgt)
#         tgt = self.tgt_pos(tgt)
#         return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
    
#     def linear(self, x):
#         return self.linear_layer(x)

