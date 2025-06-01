from transformers import BertTokenizer, BertForSequenceClassification

def download_model(model_id: str ="Charles1954/Tweet-sentiment-analysis", save_dir: str ="models/bert") -> None:
    print(f"Downloading model from Hugging Face: {model_id}")
    tokenizer = BertTokenizer.from_pretrained(model_id)
    model = BertForSequenceClassification.from_pretrained(model_id)

    print(f"Saving to: {save_dir}")
    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(save_dir)
    print("Model downloaded and saved.")

if __name__ == "__main__":
    download_model()
