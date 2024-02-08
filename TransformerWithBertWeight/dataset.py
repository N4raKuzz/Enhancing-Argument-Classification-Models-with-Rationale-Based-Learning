import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer

class TestDataset(Dataset):

    def __init__(self, texts, labels, rationales, max_len : int, tokenizer):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.rationales = rationales
        self.tokenizer = tokenizer

    def __len__(self):
        print(f"Dataset length: {len(self.texts)}")
        return len(self.texts)

    def __getitem__(self, idx):

        # Create the rationale mask for the current text
        rationales = torch.tensor(self.rationales[idx], dtype=torch.int)
        rationale_mask = self.create_rationale_mask(rationales, input_ids)

        # Encode the text
        encoding = self.tokenizer.encode(self.texts[idx])
        pad_token_id = self.tokenizer.token_to_id("[PAD]") if self.tokenizer.token_to_id("[PAD]") is not None else 0
        # Truncate
        input_ids = encoding.ids[:self.max_len]
        # Padding
        padding_length = self.max_len - len(input_ids)

        input_ids += [pad_token_id] * padding_length
        input_ids = torch.tensor(input_ids, dtype=torch.int)

        # Retrieve the label for the current text
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "label": label,
            'rationales': rationale_mask
        }
    
    def create_rationale_mask(rationales, texts):

        mask = [0] * len(texts)

        def matches_rationale(doc_index, rationale):
            if doc_index + len(rationale) > len(texts):
                return False  
            for i, token in enumerate(rationale):
                if texts[doc_index + i] != token:
                    return False
            return True

        # Iterate through each token in the document
        for i in range(len(texts)):
            for rationale in rationales:
                if matches_rationale(i, rationale):
                    for j in range(len(rationale)):
                        mask[i + j] = 1
        
        return mask