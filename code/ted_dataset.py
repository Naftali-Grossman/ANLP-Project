from torch.utils.data import Dataset
import torch
from constants import MAX_TOKENS_LENGTH

class TEDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=MAX_TOKENS_LENGTH, 
            return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transcripts = texts

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
