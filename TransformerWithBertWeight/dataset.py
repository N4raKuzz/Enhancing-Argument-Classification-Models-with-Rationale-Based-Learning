import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TestDataset(Dataset):

    def __init__(self, texts, labels, max_len : int, tokenizer):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoded_dict = self.tokenizer.encode_plus(self.texts[idx], padding = 'max_length', truncation = 'longest_first', max_length = self.max_len, return_tensors = 'pt')
        input_ids = encoded_dict['input_ids']
        label = self.labels[idx]

        return {
            "input_ids": input_ids,
            "label": label,
        }