import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleLSTM(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            embed_dim: int, 
            hidden_dim: int, 
            num_layers: int
        ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.fc.weight = self.embed.weight

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.lstm(x, hidden)
        logits = self.fc(out)
        return logits, hidden