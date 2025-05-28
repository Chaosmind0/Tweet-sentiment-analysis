import pandas as pd
import json

def JSON_transform() -> None:
    # read the csv file and create a pandas dataframe
    df = pd.read_csv("data/training.1600000.processed.noemoticon.csv", encoding="latin-1", header=None)

    # rename the columns
    df.columns = ["target", "ids", "date", "flag", "user", "text"]

    # drop the columns that are not needed
    df = df[["target", "text"]]

    # shuffle the rows
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    total = len(df)
    train_end = int(total * 0.6)
    val_end = int(total * 0.8)

    train = df.iloc[:train_end].to_dict(orient="records")
    val = df.iloc[train_end:val_end].to_dict(orient="records")
    test = df.iloc[val_end:].to_dict(orient="records")

    # create a dictionary with the train, val, and test data
    full_data = {
        "train": train,
        "val": val,
        "test": test
    }

    # save the dictionary as a JSON file
    with open("data/sentiment140_split.json", "w", encoding="utf-8") as f:
        json.dump(full_data, f, ensure_ascii=False, indent=2)

    print("Converted to JSON (split): data/sentiment140_split.json")

if __name__ == "__main__":
    JSON_transform()
