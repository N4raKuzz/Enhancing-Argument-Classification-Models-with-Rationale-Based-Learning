import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from dataset import BERTDataset
from model import BERTClassifier

MAX_LENGTH = 256
BATCH_SIZE = 16
NUM_CLASSES = 2
FILE_PATH = "data\icml_imdb_large.csv"
MODEL_PATH = "bert_classifier_imdb_large.pth"

def load_data(path):
    df = pd.read_csv(os.path.abspath(path))
    df['labels'] = df['labels'].replace(-1, 0)
    texts = df['documents'].tolist()
    labels = df['labels'].tolist()
    return texts, labels

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

# Load Data
texts, labels = load_data(FILE_PATH)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.1, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
val_dataset = BERTDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(NUM_CLASSES).to(device)
model.load_state_dict(torch.load(MODEL_PATH))

print(f"Evaluating model: {MODEL_PATH}")
accuracy, report = evaluate(model, val_dataloader, device)
print(f"Validation Accuracy: {accuracy:.4f}")
print(report)