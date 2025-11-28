import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class TextDataset(Dataset):

    def __init__(self, df: pd.DataFrame, tokenizer: AutoTokenizer) -> None:

        self.df = df
        self.tokenizer = tokenizer


    def __len__(self) -> int:
        return len(self.df)
    

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        encodings = self.tokenizer(
            row["text"],
            truncation=True,
            padding=False,
            max_length=512,
        )

        item = {
            "input_ids": torch.tensor(encodings["input_ids"]),
            "attention_mask": torch.tensor(encodings["attention_mask"]),
            "text_id": torch.tensor(idx),
            "labels": torch.tensor(row["label"] - 1)
        }

        return item