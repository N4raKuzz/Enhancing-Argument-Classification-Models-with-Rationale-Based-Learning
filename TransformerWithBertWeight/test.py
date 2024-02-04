import torch
import math

#

x = torch.randint(0, 100, (16, 1024))#.to(device) 
embedding = torch.nn.Embedding(20000, 768)#.to(device) 
print(f"Input Shpape: {x.shape}")

out = embedding(x) * math.sqrt(768)#.to(device) 
print(f"Embedding layer test(${out.shape})")

encoding = torch.zeros(1024, 768)#.to(device) 

pos = torch.arange(0, 1024, dtype=torch.float).unsqueeze(1)#.to(device) 

div_term = torch.exp(torch.arange(0, 768, 2).float() * (-math.log(10000.0) / 768)) #.to(device) 
encoding[:, 0::2] = torch.sin(pos * div_term)
encoding[:, 1::2] = torch.cos(pos * div_term)

# (1, seq_len 1024, d_model 768)
encoding = encoding.unsqueeze(0)
print(f"Encoding shape: (${out.shape})")

out = out + encoding[:, :x.shape[1], :]
print(f"Pos Encoding layer test (${out})")