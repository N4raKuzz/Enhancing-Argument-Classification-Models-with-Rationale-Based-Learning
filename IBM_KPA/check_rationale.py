# import pandas as pd
# import Levenshtein as lev

# df = pd.read_csv("data\ArgKP-2021_dataset_r.csv")
# #df['gpt_rationales'] = df['gpt_rationales'].str.replace('"', '', regex=False)

# similarities = df.apply(lambda row: lev.ratio(row['gpt_rationales'], row['rationales']), axis=1)
# average_similarity = similarities.mean()

# print(f"Average similarity of GPT Rationales and Human Raionales: {average_similarity}") # 0.6017589133008077

# # Levenshtein Ratio: This function computes the similarity as a ratio between 0 and 1, where 1 means identical strings and 0 means completely different strings. 
# # Levenshtein distance / max length of two strings


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

FILE_PATH = "data\snippet.csv"
MODEL_PATH = "model\ibm_kpa_ak_r.pth"

def load_data(path):
    df = pd.read_csv(os.path.abspath(path))
    df['stance'] = df['stance'].replace(-1, 0)
    # df['rationales'] = df['rationales'].apply(lambda x: ast.literal_eval(x))

    topics = df['topic'].tolist()
    arguments = df['argument'].tolist()
    keypoints = df['key_point'].tolist()
    labels = df['label'].tolist()
    stances = df['stance'].tolist()
    rationales = df['rationales'].tolist()

    return topics, arguments, keypoints, stances, labels, rationales

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            # Evaluating on Topic - keypoints
            # input_ids_tk = batch['input_ids_tk'].to(device)
            # targets = batch['stance'].to(device)
            # outputs = model(input_ids_tk)

            # Evaluating on Argument - keypoints
            input_ids_ak = batch['input_ids_ak'].to(device)
            targets = batch['label'].to(device)
            outputs = model(input_ids_ak)

            results = outputs[1].to(device)
            _, preds = torch.max(results, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(targets.cpu().tolist())

            if (predictions[0] != actual_labels[0]):
                print(f"Wrong Prediction with Input: {input_ids_ak} and {targets}")
                print(f"Expected {actual_labels[0]} Actual {predictions[0]}")
            # else:
            #     print(f"Correct Prediction with Input: {input_ids_ak} and {targets}")
            #     print(f"Expected {actual_labels[0]} Actual {predictions[0]}")


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load Data
topics, arguments, keypoints, stances, labels, rationales = load_data(FILE_PATH)
train_topics, val_topics, train_arguments, val_arguments, train_keypoints, val_keypoints, train_stances, val_stances, train_labels, val_labels, train_rationales, val_rationales = train_test_split(topics, arguments, keypoints, stances, labels, rationales, test_size=0.2, random_state=42)

# Initialize dataset
val_dataset = RationaleDataset(topics, arguments, keypoints, stances, labels, rationales, MAX_LEN, tokenizer)
print(f"Total validation samples: {len(val_dataset)}")

# Initializer Dataloader
val_dataloader = DataLoader(val_dataset, batch_size=1)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
input_embedding = InputEmbedding(D_MODEL, V_SIZE).to(device) 
positional_encoding = PositionalEncoding(D_MODEL, MAX_LEN, DROPOUT,device).to(device)
model = TransformerBertWeight(NUM_CLASSES, D_MODEL, D_FF, input_embedding, positional_encoding, num_heads = NUM_HEADS, dropout = DROPOUT).to(device)
model.load_state_dict(torch.load(MODEL_PATH)) #, map_location=torch.device('cpu')))

print(f"Evaluating model: {MODEL_PATH}")
evaluate(model, val_dataloader, device)