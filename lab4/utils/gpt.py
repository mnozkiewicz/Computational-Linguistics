import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint # <--- 1. Import Checkpoint
from flash_attn.flash_attn_interface import flash_attn_func


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, context_length=128):
        super().__init__()

        pe = torch.zeros(context_length, embed_dim)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float()
            * (-torch.log(torch.tensor(10000.0)) / embed_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        window: tuple[int, int] = (-1, -1)
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

        self.window = window
        self.attn_forward = self.flash_attention_forward if use_flash_attention else self.attenntion_forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.num_heads, self.head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim)

        out = self.attn_forward(q, k, v, T)
        out = out.contiguous().view(B, T, C)
        
        return self.out_proj(out)


    def flash_attention_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, T: int) -> torch.Tensor:
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=self.window
        )
        return out

    def attenntion_forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, T: int) -> torch.Tensor:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.tril(torch.ones(T, T, device=q.device))
        attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)

        out = attn_probs @ v
        out = out.transpose(1, 2)
        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        window: tuple[int, int] = (-1, -1)
    ):
        super().__init__()

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(
            embed_dim,
            num_heads,
            dropout,
            use_flash_attention=use_flash_attention,
            window=window
        )
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = FeedForward(embed_dim, ff_hidden_dim, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        ff_hidden_dim: int,
        num_layers: int,
        context_length: int,
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        window: tuple[int, int] = (-1, -1)
    ):
        super().__init__()

        self.use_flash_attention = use_flash_attention
        # 2. Initialize Checkpointing Flag
        self.gradient_checkpointing = False
        
        self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(
            embed_dim, context_length
        )

        self.layers = nn.ModuleList([
            DecoderBlock(
                embed_dim,
                num_heads,
                ff_hidden_dim,
                dropout,
                use_flash_attention=use_flash_attention,
                window=window
            )
            for _ in range(num_layers)
        ])

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.embed_tokens.weight

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing = True

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        x = self.pos_encoding(x)


        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.ln_f(x)
        return self.lm_head(x)