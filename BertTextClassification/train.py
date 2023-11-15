import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from dataset import BERTDataset
from model import BERTClassifier
from sklearn.model_selection import train_test_split

FILE_PATH = "\data\icml_imdb_large.csv"
NUM_CLASS = 2
MAX_LENGTH = 128
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE= 2e-5

def load_data(path):
    df = pd.read_csv(path)
    texts = df['documents'].tolist()
    labels = df['labels'].tolist()
    return texts, labels
    
texts, labels = load_data(FILE_PATH)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained("imdb_small")
train_dataset = BERTDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
val_dataset = BERTDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier("imdb_small", NUM_CLASS).to(device)

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    model.train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
        print(report)


