import torch
import os
import pandas as pd
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from model import TransformerBertWeight, InputEmbedding, PositionalEncoding
from transformers import BertTokenizer
from dataset import TestDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Model parameters
FILE_PATH = "data\icml_test.csv"
NUM_CLASSES = 2  # Number of classes
D_MODEL = 768  # Model dimension
D_FF = 2048  # Dimension of feed-forward network
NUM_HEADS = 8  # Number of attention heads
MAX_LEN = 1024  # Maximum sequence length
V_SIZE = 20000  # Size of vocabulary
LEARNING_RATE = 1e-3 # Learning Rate
NUM_EPOCHS = 1 # Number of epochs
DROPOUT = 0.1 

def load_data(path):
    df = pd.read_csv(os.path.abspath(path))
    df['labels'] = df['labels'].replace(-1, 0)
    texts = df['documents'].tolist()
    labels = df['labels'].tolist()
    return texts, labels

def evaluate(model, data_loader, device):
    model.eval().to(device)
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)  
            labels = batch['label'].to(device)
            outputs = model(input_ids).to(device)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
            
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def train(model, data_loader, optimizer, device):
    model.train().to(device)
    for batch in data_loader:
        print(f"Batch size: {len(batch['input_ids'])}")
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids).to(device)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()   

# Load Data
texts, labels = load_data(FILE_PATH)
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Initialize device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
if (device == 'cuda'):
    print(f"Device name: {torch.cuda.get_device_name(device.index)}")
    print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

# Initialize tokenizer
input_embedding = InputEmbedding(D_MODEL, V_SIZE).to(device) 
positional_encoding = PositionalEncoding(D_MODEL, MAX_LEN, DROPOUT, device).to(device) 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Initialize dataset
train_dataset = TestDataset(train_texts, train_labels, MAX_LEN, tokenizer)
val_dataset = TestDataset(val_texts, val_labels, MAX_LEN, tokenizer)
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")

# Initialize model
model = TransformerBertWeight(NUM_CLASSES, D_MODEL, D_FF, input_embedding, positional_encoding, device, num_heads = NUM_HEADS, dropout = DROPOUT)

# Initializer Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

# Training Loop
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    train(model, train_dataloader, optimizer, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)

# Save model
torch.save(model.state_dict(), 'encoder_classfier_with_bert_weight_imdb_small.pth')