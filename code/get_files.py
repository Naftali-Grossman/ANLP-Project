from huggingface_hub import hf_hub_download
import pandas as pd
import pickle


def get_dataset(filename):
    file_path = hf_hub_download(
    repo_id="Naftali1996/Datasets",
    filename=filename,
    repo_type="dataset"
)
    if filename.endswith(".csv"):
        return pd.read_csv(file_path)
    elif filename.endswith(".json"):
        return pd.read_json(file_path)
    elif filename.endswith(".pkl"):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")
    
    