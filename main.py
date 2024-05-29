import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

# Load the dataset
dataset = load_dataset("opus100", "en-fr")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("t5-base")

# Preprocess the dataset
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, truncation=True, padding=True)
    labels = tokenizer(targets, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Custom collate function to handle padding
def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    
    input_ids = tokenizer.pad({"input_ids": input_ids}, return_tensors="pt")["input_ids"]
    labels = tokenizer.pad({"input_ids": labels}, return_tensors="pt")["input_ids"]
    
    return {"input_ids": input_ids, "labels": labels}

# Define the translation model
class TranslationModel(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, hidden_dim, num_layers):
        super(TranslationModel, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embedding_dim)
        self.encoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.decoder = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_vocab_size)
        
    def forward(self, input_seq, target_seq):
        input_embedded = self.embedding(input_seq)
        _, (hidden, cell) = self.encoder(input_embedded)
        
        target_embedded = self.embedding(target_seq)
        output, _ = self.decoder(target_embedded, (hidden, cell))
        output = self.fc(output)
        
        return output

# Initialize model and training parameters
input_vocab_size = len(tokenizer)
output_vocab_size = len(tokenizer)
embedding_dim = 256
hidden_dim = 512
num_layers = 2
learning_rate = 0.001
num_epochs = 10
batch_size = 32
grad_accum_steps = 2  # Accumulate gradients over this many steps

# Determine the device to use (CUDA, MPS for M1 Macs, or CPU)
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = TranslationModel(input_vocab_size, output_vocab_size, embedding_dim, hidden_dim, num_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define data loaders with custom collate function
train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=batch_size, collate_fn=collate_fn)

# Training function
def train(model, loader, criterion, optimizer, device, grad_accum_steps):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()
    for i, batch in enumerate(tqdm(loader, desc="Training", leave=False)):
        input_seq = batch["input_ids"].to(device)
        target_seq = batch["labels"].to(device)
        
        output = model(input_seq, target_seq[:, :-1])
        
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        target = target_seq[:, 1:].contiguous().view(-1)
        
        loss = criterion(output, target)
        loss.backward()
        
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

# Evaluation function
def evaluate(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_seq = batch["input_ids"].to(device)
            target_seq = batch["labels"].to(device)
            
            output = model(input_seq, target_seq[:, :-1])
            
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            target = target_seq[:, 1:].contiguous().view(-1)
            
            loss = criterion(output, target)
            epoch_loss += loss.item()
    
    return epoch_loss / len(loader)

# Training loop with mixed precision
scaler = torch.cuda.amp.GradScaler()

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device, grad_accum_steps)
    val_loss = evaluate(model, val_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

# Save the trained model
torch.save(model.state_dict(), "translation_model.pth")
