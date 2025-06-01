from train_model import train_model
import json

def run_training() -> None:
    # load data from json file
    with open("data/sentiment140_split.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    # label conversion function
    label_map = {0: 0, 4: 1}

    # convert labels to 0 or 1
    def convert_label(label):
        return label_map[int(label)]

    # extract data from json file
    train_texts = [item["text"] for item in data["train"]]
    train_labels = [convert_label(item["target"]) for item in data["train"]]

    val_texts = [item["text"] for item in data["val"]]
    val_labels = [convert_label(item["target"]) for item in data["val"]]

    # start training
    train_model(
        train_texts, train_labels,  #  use training set for training
        val_texts, val_labels,      #  use validation set for evaluation
        model_name="bert",          #  use BERT model
        epochs=8,                   #  train for 3 epochs
        batch_size=16,              #  batch size of 16
        learning_rate=2e-5,         #  learning rate of 2e-5
        save_dir="models/bert"      #  save model to "models/bert"
    )

if __name__ == "__main__":
    run_training()