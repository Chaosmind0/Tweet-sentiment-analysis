import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
)
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm
from torch.optim import AdamW


class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def train_model(
    train_texts, train_labels,
    val_texts, val_labels,
    model_name="bert", epochs=3, batch_size=16, learning_rate=2e-5,
    save_dir="models/bert"
):
    assert model_name in ["bert", "distilbert"], "Only 'bert' or 'distilbert' supported"
    os.makedirs(save_dir, exist_ok=True)

    # Check if saved model exists
    if os.path.exists(os.path.join(save_dir, "pytorch_model.bin")):
        print("Found saved model. Loading...")
        if model_name == "bert":
            tokenizer = BertTokenizer.from_pretrained(save_dir)
            model = BertForSequenceClassification.from_pretrained(save_dir)
        else:
            tokenizer = DistilBertTokenizer.from_pretrained(save_dir)
            model = DistilBertForSequenceClassification.from_pretrained(save_dir)
    else:
        print("No saved model found. Training from scratch...")
        if model_name == "bert":
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
        else:
            tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
            model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Prepare dataset and dataloader
        train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
        val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        optimizer = AdamW(model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            model.train()
            total_loss = 0

            for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        # Save model and tokenizer
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Model saved to: {save_dir}")

    # Evaluate on validation set
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    preds, true = [], []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
            outputs = model(**inputs)
            logits = outputs.logits
            preds += torch.argmax(logits, dim=1).cpu().tolist()
            true += batch["labels"].tolist()

    acc = accuracy_score(true, preds)
    f1 = f1_score(true, preds, average="weighted")

    print(f"Evaluation Accuracy: {acc:.4f}")
    print(f"Evaluation F1 Score: {f1:.4f}")

    return model
