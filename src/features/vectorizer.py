from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from transformers import BertTokenizer, BertModel
import torch
import numpy as np

class TextVectorizer:
    # Constructor
    def __init__(self, method="tfidf"):
        self.method = method.lower()    # Convert to lowercase for consistency
        self.vectorizer = None          # Vectorizer object
        self.tokenizer = None           # BERT tokenizer object
        self.model = None               # BERT model object


        if self.method == "count":
            self.vectorizer = CountVectorizer()  # Initialize CountVectorizer
        elif self.method == "tfidf":
            self.vectorizer = TfidfVectorizer()  # Initialize TfidfVectorizer
        elif self.method == "bert":
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased") 
            self.model = BertModel.from_pretrained("bert-base-uncased")
            self.model.eval()
        else:
            raise ValueError(f"Unsupported method: {self.method}")

    # fit_transform and transform methods are the same for all vectorizers
    def fit_transform(self, texts):
        if self.method in ["count", "tfidf"]:
            return self.vectorizer.fit_transform(texts)
        elif self.method == "bert":
            return self._bert_encode(texts)

    # transform method is the same for all vectorizers
    def transform(self, texts):
        if self.method in ["count", "tfidf"]:
            return self.vectorizer.transform(texts)
        elif self.method == "bert":
            return self._bert_encode(texts)

    # _bert_encode method is specific to BERT vectorizer
    def _bert_encode(self, texts):
        embeddings = []
        with torch.no_grad():
            for text in texts:
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
                outputs = self.model(**inputs)
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token embedding
                embeddings.append(cls_embedding.squeeze(0).numpy())
        return np.array(embeddings)
