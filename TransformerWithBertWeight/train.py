import torch
import os
import ast
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import Adam
from model import TransformerBertWeight, InputEmbedding, PositionalEncoding
from transformers import BertTokenizer
from dataset import RationaleDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Model parameters
FILE_PATH = "data\snippet.csv"
TOKENIZER_PATH = "tokenizer.json"
NUM_CLASSES = 2  # Number of classes
D_MODEL = 768  # Model dimension
D_FF = 2048  # Dimension of feed-forward network
NUM_HEADS = 8  # Number of attention heads
MAX_LEN = 1024  # Maximum sequence length
V_SIZE = 30522  # Size of vocabulary
LEARNING_RATE = 1e-3 # Learning Rate
NUM_EPOCHS = 1 # Number of epochs
DROPOUT = 0.1 
LAMBDA = 1

def load_data(path):
    df = pd.read_csv(os.path.abspath(path))
    df['labels'] = df['labels'].replace(-1, 0)
    df['rationales'] = df['rationales'].apply(lambda x: ast.literal_eval(x))

    texts = df['documents'].tolist()
    labels = df['labels'].tolist()
    rationales = df['rationales'].tolist()
    print(labels)
    return texts, labels, rationales

def evaluate(model, data_loader, device):
    model.eval()
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
    model.train()
    for batch in data_loader:
        print(f"Batch size: {len(batch['input_ids'])}")
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        #rationale = batch['rationale'].to(device)

        outputs = model(input_ids).to(device)
        #outputs, att_scrores = model(input_ids).to(device)
        loss = model.loss_cross_entropy_softmax(outputs, labels).to(device) #* (1-LAMBDA) + LAMBDA * model.loss_attention_rationales(att_scrores, rationale)
        loss.backward()
        optimizer.step()   

# Load Data
texts, labels, rationales = load_data(FILE_PATH)
train_texts, val_texts, train_labels, val_labels, train_rationales, val_rationales = train_test_split(texts, labels, rationales, test_size=0.2, random_state=42)

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
train_dataset = RationaleDataset(train_texts, train_labels, train_rationales, MAX_LEN, tokenizer)
val_dataset = RationaleDataset(val_texts, val_labels, val_rationales, MAX_LEN, tokenizer)
print(f"Total training samples: {len(train_dataset)}")
print(f"Total validation samples: {len(val_dataset)}")

# Initialize model
model = TransformerBertWeight(NUM_CLASSES, D_MODEL, D_FF, input_embedding, positional_encoding, num_heads = NUM_HEADS, dropout = DROPOUT).to(device) 

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
torch.save(model.state_dict(), 'model\snippet.pth')