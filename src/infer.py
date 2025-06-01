import torch
from transformers import BertTokenizer, BertForSequenceClassification

def predict_sentiment(text: str, model_dir: str ="models/bert") -> str:
    # load local tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # tokenize input text
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # predict sentiment
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred_id = torch.argmax(logits, dim=1).item()

    # map id to label（0=negative，1=positive）
    label_map = {0: "Negative", 1: "Positive"}
    return label_map.get(pred_id, "Unknown")

# Example usage:
if __name__ == "__main__":
    sample_text = "I would like to have some pizza after I finish this project."
    sentiment = predict_sentiment(sample_text)
    print(f"Text: {sample_text}")
    print(f"Predicted Sentiment: {sentiment}")
