import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

class RationaleDataset(Dataset):

    def __init__(self, topics, arguments, keypoints, stances, labels, rationales, max_len : int, tokenizer):
        self.topics = topics
        self.arguments = arguments
        self.keypoints = keypoints
        self.stances = stances
        self.labels = labels
        self.max_len = max_len
        self.rationales = rationales
        self.tokenizer = tokenizer

    def __len__(self):
        # print(f"Dataset length: {len(self.texts)}")
        return len(self.topics)

    def __getitem__(self, idx):

        # Tokenize rationales
        rationale = self.rationales[idx]
        if not isinstance(rationale, str):
            rationale = str(rationale)  # Convert non-string to string
        rationale_token = self.tokenizer.tokenize(rationale)
        rationale_token = self.tokenizer(rationale)

        # Encode the topic-keypoint and the argument-keypoint pair
        encoding  = self.tokenizer(self.topics[idx] + "[SEP]" + self.keypoints[idx], padding="max_length", max_length=self.max_len, truncation=True)
        input_ids_topic = encoding['input_ids']
        encoding  = self.tokenizer(self.arguments[idx] + "[SEP]" + self.keypoints[idx], padding="max_length", max_length=self.max_len, truncation=True)
        input_ids_argument = encoding['input_ids']
        # print(f"text : {torch.tensor(input_ids, dtype=torch.int64)}")

        #Padding and Truncation of rationales masl for topic-keypoint and the argument-keypoint pair
        rationale_mask_tk = self.create_rationale_mask(rationale_token, input_ids_topic)
        rationale_mask_tk = rationale_mask_tk[:self.max_len] if len(rationale_mask_tk) > self.max_len else rationale_mask_tk
        rationale_mask_tk = np.array([rationale_mask_tk + [0 for _ in range(self.max_len - len(rationale_mask_tk))]])
        rationale_mask_tk = torch.tensor(rationale_mask_tk, dtype=torch.int32)

        rationale_mask_ak = self.create_rationale_mask(rationale_token, input_ids_argument)
        rationale_mask_ak = rationale_mask_ak[:self.max_len] if len(rationale_mask_ak) > self.max_len else rationale_mask_ak
        rationale_mask_ak = np.array([rationale_mask_ak + [0 for _ in range(self.max_len - len(rationale_mask_ak))]])
        rationale_mask_ak = torch.tensor(rationale_mask_ak, dtype=torch.int32)
        # print(f"Shape of rationale: {rationale_mask.shape}")

        return {
            "input_ids_tk": torch.tensor(input_ids_topic, dtype=torch.int64),
            "input_ids_ak": torch.tensor(input_ids_argument, dtype=torch.int64),
            "stance": torch.tensor(self.stances[idx], dtype=torch.int64),
            "label": torch.tensor(self.labels[idx], dtype=torch.int64),
            "rationales_tk": rationale_mask_tk,
            "rationales_ak": rationale_mask_ak
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