import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class RationaleDataset(Dataset):

    def __init__(self, texts, labels, rationales, max_len : int, tokenizer):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        self.rationales = rationales
        self.tokenizer = tokenizer

    def __len__(self):
        # print(f"Dataset length: {len(self.texts)}")
        return len(self.texts)

    def __getitem__(self, idx):

        # Create the rationale mask for the current text
        # print(f"rationale : {self.rationales[idx]} \n")
        # print(f"text : {self.texts[idx]}")
        rationale_mask = self.create_rationale_mask(self.rationales[idx], self.texts[idx])

        #Padding and Truncation of rationales
        rationale_mask = rationale_mask[:self.max_len] if len(rationale_mask) > self.max_len else rationale_mask
        rationale_mask = np.array([rationale_mask + [0 for _ in range(self.max_len - len(rationale_mask))]])
        rationale_mask = torch.tensor(rationale_mask, dtype=torch.int32)
        # print(f"Shape of rationale: {rationale_mask.shape}")

        # Encode the text
        encoding  = self.tokenizer(self.texts[idx], padding="max_length", max_length=self.max_len, truncation=True)
        input_ids = encoding['input_ids']
        # print(f"text : {torch.tensor(input_ids, dtype=torch.int64)}")

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "label": torch.tensor(self.labels[idx], dtype=torch.int64),
            "rationales": rationale_mask
        }
    
    def create_rationale_mask(self, rationales, texts):

        mask = [1] * len(texts)

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
                        mask[i + j] = -1
        
        return mask