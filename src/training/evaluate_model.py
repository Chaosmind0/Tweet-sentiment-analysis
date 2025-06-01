import json
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from transformers import BertTokenizer, BertForSequenceClassification
from tqdm import tqdm

def evaluate_model(model_dir : str, data_path : str, report_dir : str, test_data : str) -> None:
    # Routes configuration
    MODEL_DIR = model_dir
    DATA_PATH = data_path
    REPORT_DIR = report_dir
    os.makedirs(REPORT_DIR, exist_ok=True)

    # Load data
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    data_set = data[test_data]
    texts = [item["text"] for item in data_set]
    label_map = {0: 0, 4: 1}
    labels = [label_map[int(item["target"])] for item in data_set]

    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # Batch reasoning
    preds = []
    batch_size = 32
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.extend(batch_preds)

    # Save the classification report.
    report = classification_report(labels, preds, target_names=["negative", "positive"])
    with open(os.path.join(REPORT_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    # Print classification report
    print(report)

    # Visualization of confusion matrix
    cm = confusion_matrix(labels, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["neg", "pos"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"))
    plt.close()

    # Save the predictions to a CSV file
    pred_label_map = {0: "negative", 1: "neutral", 2: "positive"}
    df = pd.DataFrame({
        "text": texts,
        "true_label": [pred_label_map[l] for l in labels],
        "predicted_label": [pred_label_map[p] for p in preds]
    })
    df.to_csv(os.path.join(REPORT_DIR, "predictions.csv"), index=False)

    print("Evaluation results saved to " + REPORT_DIR)

if __name__ == "__main__":
    evaluate_model("models/bert", "data/sentiment140_split.json", "report/val", "val")
    evaluate_model("models/bert", "data/sentiment140_split.json", "report/test", "test")
