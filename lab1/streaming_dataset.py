import torch
from torch.utils.data import IterableDataset
from typing import Generator
from transformers import AutoTokenizer

class StreamingTokenDataset(IterableDataset):
    def __init__(
            self, 
            dataset: IterableDataset,
            tokenizer: AutoTokenizer,
            context_size=128, 
            buffer_size=10_000
        ) -> None:

        self.dataset = dataset
        self.tokenizer = tokenizer

        self.context_size = context_size
        self.buffer_size = buffer_size
        self.sep_token_id = self.tokenizer.sep_token_id

    def _token_stream(self) -> Generator[int, None, None]:
        for example in self.dataset:
            tokens = self.tokenizer.encode(example["text"], add_special_tokens=False)
            yield from tokens
            yield self.sep_token_id

    def _chunk_stream(self):
        buf = []
        for token in self._token_stream():
            buf.append(token)
            if len(buf) > self.context_size:

                context_batch = buf[:self.context_size + 1]

                input_tokens = torch.tensor(context_batch[:self.context_size], dtype=torch.long)
                pred_tokens = torch.tensor(context_batch[1:], dtype=torch.long)
                yield input_tokens, pred_tokens
                buf = buf[self.context_size:]

    def __iter__(self):
        yield from self._chunk_stream()

