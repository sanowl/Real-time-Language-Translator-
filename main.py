import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR

# Load the dataset
dataset = load_dataset("opus100", "en-fr")

# Load the tokenizer and model
model_name = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
model = MBartForConditionalGeneration.from_pretrained(model_name)

# Set source and target languages
tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "fr_XX"

# Preprocess the dataset
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True, padding="max_length")
    return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=dataset["train"].column_names)

# Define data loaders
train_loader = DataLoader(tokenized_datasets["train"], batch_size=16, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=16)

# Set up training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=10)
num_epochs = 10

# Training function
def train(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    return total_loss / len(loader)

# Evaluation function
def evaluate(model, loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    return total_loss / len(loader)

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, scheduler, device)
    val_loss = evaluate(model, val_loader, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "best_translation_model.pth")

# Load the best model for inference
model.load_state_dict(torch.load("best_translation_model.pth"))

# Inference function with beam search
def translate(text, model, tokenizer, device, num_beams=4, max_length=50):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt").to(device)
    translated = model.generate(**inputs, num_beams=num_beams, max_length=max_length)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Example usage
text = "Hello, how are you?"
translated = translate(text, model, tokenizer, device)
print(f"Original: {text}")
print(f"Translated: {translated}")
