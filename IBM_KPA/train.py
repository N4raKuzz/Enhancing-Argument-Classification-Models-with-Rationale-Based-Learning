import torch
import os
import ast
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import TransformerBertWeight, InputEmbedding, PositionalEncoding
from transformers import BertTokenizer
from dataset import RationaleDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Model parameters
FILE_PATH = "data\ArgKP-2021_dataset_r.csv"
NUM_CLASSES = 2  # Number of classes
D_MODEL = 768  # Model dimension
D_FF = 2048  # Dimension of feed-forward network
NUM_HEADS = 8  # Number of attention heads
MAX_LEN = 128  # Maximum sequence length
V_SIZE = 30522  # Size of vocabulary
LEARNING_RATE = 1e-5 # Learning Rate
NUM_EPOCHS = 10 # Number of epochs
DROPOUT = 0.1 
LAMBDA = 0.8

def load_data(path):
    df = pd.read_csv(os.path.abspath(path))
    df['stance'] = df['stance'].replace(-1, 0)
    # df['rationales'] = df['rationales'].apply(lambda x: ast.literal_eval(x))

    topics = df['topic'].tolist()
    arguments = df['argument'].tolist()
    keypoints = df['key_point'].tolist()
    labels = df['label'].tolist()
    stances = df['stance'].tolist()
    rationales = df['gpt_rationales'].tolist()

    return topics, arguments, keypoints, stances, labels, rationales

#TODO: rewrite this
def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            # Evaluating on Topic - keypoints
            input_ids_tk = batch['input_ids_tk'].to(device)
            targets = batch['stance'].to(device)
            outputs = model(input_ids_tk)

            # Evaluating on Argument - keypoints
            # input_ids_ak = batch['input_ids_ak'].to(device)
            # targets = batch['label'].to(device)
            # outputs = model(input_ids_ak)

            results = outputs[1].to(device)
            _, preds = torch.max(results, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(targets.cpu().tolist())
            
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions, zero_division=0)

def train(model, data_loader, optimizer, device):
    model.train()
    model = model.to(device)

    for batch in data_loader:
        # print(f"Batch size: {len(batch['input_ids'])}")
        optimizer.zero_grad()
        # Training on Topic - keypoints
        input_ids_tk = batch['input_ids_tk'].to(device)
        targets = batch['stance'].to(device)
        outputs = model(input_ids_tk)
        rationales = batch['rationales_tk'].to(device)

        # Training on Argument - keypoints
        # input_ids_ak = batch['input_ids_ak'].to(device)
        # targets = batch['label'].to(device)
        # outputs = model(input_ids_ak)
        # rationales = batch['rationales_ak'].to(device)
        
        att_scrores = outputs[0].to(device)

        results = outputs[1].to(device)
        CrossEntropyLoss = nn.CrossEntropyLoss()
        # loss = CrossEntropyLoss(results, targets)
        loss = LAMBDA * CrossEntropyLoss(results, targets) + (1-LAMBDA) * model.attention_loss(att_scrores, rationales)
        # print(f"Total loss: {loss}")
        # loss_trend.append(loss.detach().cpu().numpy())
        loss.backward()
        optimizer.step()   

# Load Data
topics, arguments, keypoints, stances, labels, rationales = load_data(FILE_PATH)
train_topics, val_topics, train_arguments, val_arguments, train_keypoints, val_keypoints, train_stances, val_stances, train_labels, val_labels, train_rationales, val_rationales = train_test_split(topics, arguments, keypoints, stances, labels, rationales, test_size=0.2, random_state=42)

# Initialize device
device = torch.device("cuda")# if torch.cuda.is_initialized() else "cpu")
#device = torch.device("cpu")
print(f"Device name: {torch.cuda.get_device_name(device.index)}")
print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")


# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
# tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
# tokenizer.normalizer = normalizers.Sequence(
#     [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()]
# )
# tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
# trainer = trainers.WordPieceTrainer(vocab_size = V_SIZE, special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SOS]", "[EOS]"])
# tokenizer.train_from_iterator(texts, trainer = trainer)
# tokenizer.save(TOKENIZER_PATH)

input_embedding = InputEmbedding(D_MODEL, V_SIZE).to(device) 
positional_encoding = PositionalEncoding(D_MODEL, MAX_LEN, DROPOUT,device).to(device)

# Initialize dataset
train_dataset = RationaleDataset(train_topics, train_arguments, train_keypoints, train_stances, train_labels, train_rationales, MAX_LEN, tokenizer)
val_dataset = RationaleDataset(val_topics, val_arguments, val_keypoints, val_labels, val_stances, val_rationales, MAX_LEN, tokenizer)
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")

# Initialize model
model = TransformerBertWeight(NUM_CLASSES, D_MODEL, D_FF, input_embedding, positional_encoding, num_heads = NUM_HEADS, dropout = DROPOUT).to(device) 

# Initializer Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4)

optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

loss_trend = []
# Training Loop
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch + 1} / {NUM_EPOCHS}")
    loss_trend = []
    train(model, train_dataloader, optimizer, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)

# Save model
torch.save(model.state_dict(), 'model\snippet.pth')