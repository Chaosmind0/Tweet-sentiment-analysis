from train_model import train_model
import json

# load data from json file
with open("data/sentiment140_split.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# label conversion function
label_map = {0: 0, 2: 1, 4: 2}

def convert_label(label):
    return label_map[int(label)]

train_texts = [item["text"] for item in data["train"]]
train_labels = [convert_label(item["target"]) for item in data["train"]]

val_texts = [item["text"] for item in data["val"]]
val_labels = [convert_label(item["target"]) for item in data["val"]]

# start training
train_model(
    train_texts, train_labels,
    val_texts, val_labels,
    model_name="bert",     
    epochs=3,
    batch_size=16,
    learning_rate=2e-5,
    save_dir="models/bert"  #  save model to "models/bert"
)
