import os
import shutil
import kagglehub

def setup_kaggle_api_key() -> None:
    src = "kaggle.json"
    dst_dir = os.path.expanduser("~/.kaggle")
    dst = os.path.join(dst_dir, "kaggle.json")

    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(src, dst)
        print("kaggle.json moved to ~/.kaggle/")
    else:
        print("kaggle API key already configured.")

def move_files_to_data_folder(src_path) -> None:
    os.makedirs("data", exist_ok=True)
    for file in os.listdir(src_path):
        full_src = os.path.join(src_path, file)
        full_dst = os.path.join("data", file)
        if not os.path.exists(full_dst):
            shutil.copy(full_src, full_dst)
    print("Files moved to ./data/")

def download_dataset() -> None:
    print("Downloading Sentiment140...")
    path = kagglehub.dataset_download("kazanova/sentiment140")
    print("Dataset downloaded to:", path)
    move_files_to_data_folder(path)

if __name__ == "__main__":
    setup_kaggle_api_key()
    download_dataset()