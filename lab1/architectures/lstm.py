import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SimpleLSTM(nn.Module):
    def __init__(
            self, 
            vocab_size: int, 
            embed_dim: int, 
            hidden_dim: int, 
            num_layers: int,
            embedding: Optional[nn.Embedding] = None
        ):
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.fc.weight = self.embed.weight

    def forward(self, x, hidden=None):
        x = self.embed(x)            # [batch, seq_len, embed_dim]
        out, hidden = self.lstm(x, hidden)  # [batch, seq_len, hidden_dim]
        logits = self.fc(out)        # [batch, seq_len, vocab_size]
        return logits, hidden