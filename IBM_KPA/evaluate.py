import torch
import os
import ast
import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from dataset import RationaleDataset
from model import TransformerBertWeight, InputEmbedding, PositionalEncoding

# Model parameters
NUM_CLASSES = 2  # Number of classes
D_MODEL = 768  # Model dimension
D_FF = 2048  # Dimension of feed-forward network
NUM_HEADS = 8  # Number of attention heads
MAX_LEN = 1024  # Maximum sequence length
V_SIZE = 30522  # Size of vocabulary
DROPOUT = 0.1 

FILE_PATH = "data\ArgKP-2021_dataset.csv"
MODEL_PATH = "model\ibm_kpa_tk_r.pth"
# FILE_PATH = "data\icml_imdb_large.csv"
# MODEL_PATH = "model\imdb_large_r2.pth"

def load_data(path):
    df = pd.read_csv(os.path.abspath(path))
    df['stance'] = df['stance'].replace(-1, 0)
    # df['rationales'] = df['rationales'].apply(lambda x: ast.literal_eval(x))

    topics = df['topic'].tolist()
    arguments = df['argument'].tolist()
    keypoints = df['key_point'].tolist()
    labels = df['label'].tolist()
    stances = df['label'].tolist()
    rationales = df['rationales'].tolist()

    return topics, arguments, keypoints, stances, labels, rationales

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

    accuracy, precision, recall, f1_score = calculate_metrics(actual_labels, predictions)
    print_classification_report(accuracy, precision, recall, f1_score)

def calculate_metrics(y_true, y_pred):
    unique_classes = set(y_true)
    true_positives = dict.fromkeys(unique_classes, 0)
    false_positives = dict.fromkeys(unique_classes, 0)
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            true_positives[true] += 1
        else:
            false_positives[pred] += 1
    
    precision = {}
    recall = {}
    f1_score = {}
    
    for cls in unique_classes:
        precision[cls] = true_positives[cls] / (true_positives[cls] + false_positives[cls]) if true_positives[cls] + false_positives[cls] > 0 else 0
        recall[cls] = true_positives[cls] / (true_positives[cls]) if true_positives[cls] > 0 else 0
        if precision[cls] + recall[cls] > 0:
            f1_score[cls] = 2 * precision[cls] * recall[cls] / (precision[cls] + recall[cls])
        else:
            f1_score[cls] = 0
    
    return accuracy, precision, recall, f1_score

def print_classification_report(accuracy, precision, recall, f1_score):
    print(f"Accuracy: {accuracy:.2f}")
    print("Class\tPrecision\tRecall\tF1-Score")
    for cls in precision:
        print(f"{cls}\t{precision[cls]:.2f}\t\t{recall[cls]:.2f}\t{f1_score[cls]:.2f}")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load Data
topics, arguments, keypoints, stances, labels, rationales = load_data(FILE_PATH)
train_topics, val_topics, train_arguments, val_arguments, train_keypoints, val_keypoints, train_stances, val_stances, train_labels, val_labels, train_rationales, val_rationales = train_test_split(topics, arguments, keypoints, stances, labels, rationales, test_size=0.2, random_state=42)

# Initialize dataset
train_dataset = RationaleDataset(train_topics, train_arguments, train_keypoints, train_stances, train_labels, train_rationales, MAX_LEN, tokenizer)
val_dataset = RationaleDataset(val_topics, val_arguments, val_keypoints, val_labels, val_stances, val_rationales, MAX_LEN, tokenizer)
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")

# Initializer Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
input_embedding = InputEmbedding(D_MODEL, V_SIZE).to(device) 
positional_encoding = PositionalEncoding(D_MODEL, MAX_LEN, DROPOUT,device).to(device)
model = TransformerBertWeight(NUM_CLASSES, D_MODEL, D_FF, input_embedding, positional_encoding, num_heads = NUM_HEADS, dropout = DROPOUT).to(device)
model.load_state_dict(torch.load(MODEL_PATH)) #, map_location=torch.device('cpu')))

print(f"Evaluating model: {MODEL_PATH}")
evaluate(model, val_dataloader, device)