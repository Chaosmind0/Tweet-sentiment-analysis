import pandas as pd
import json
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# download required packages
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_text(text: str) -> str:

    text = re.sub(r"http\S+|www\S+|@\w+|#\w+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation)).lower()

    tokens = text.split()

    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return " ".join(tokens)


def JSON_transform(ignore_rate: float) -> None:
    # read CSV file
    df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)
    df.columns = ["target", "ids", "date", "flag", "user", "text"]
    df = df[["target", "text"]]

    # clean text
    print("Cleaning text...")
    df["text"] = df["text"].apply(clean_text)

    # split data
    total = len(df) * (1 - ignore_rate)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:int(total)-1].copy()

    # clean text in training set
    print("Cleaning text in training set...")
    train_df["text"] = train_df["text"].apply(clean_text)

    full_data = {
        "train": train_df.to_dict(orient="records"),
        "val": val_df.to_dict(orient="records"),
        "test": test_df.to_dict(orient="records")
    }

    # save as JSON
    with open("data/sentiment140_split.json", "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    print("Converted & cleaned JSON saved to: data/sentiment140_split.json")

if __name__ == "__main__":
    ignore_rate = 0.99   # ignore 99% of data for dataset size reduction
    JSON_transform(ignore_rate)
