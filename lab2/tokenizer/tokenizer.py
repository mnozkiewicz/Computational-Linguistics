from typing import Optional, Protocol
from transformers import AutoTokenizer
import sentencepiece as spm
from collections import Counter
import json
import re



class Tokenizer(Protocol):

    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, tokens: list[int]) -> str:
        ...

    def vocab_size(self) -> int:
        ...


class WPTokenizer:
    def __init__(self, vocab_size: int, unk_token="<UNK>", word2id: Optional[dict[str, int]] = None):
        self._vocab_size = vocab_size
        self.unk_token = unk_token

        if word2id is None:
            self.word2id = {}
            self.id2word = {}
        else:
            self.word2id = word2id
            self.id2word = {i: w for w, i in self.word2id.items()}

    def build_vocab(self, texts: list[str]):
        counter = Counter()
        for text in texts:
            text = text.lower()
            tokens = self._tokenize(text)
            counter.update(tokens)
        
        most_common = counter.most_common(self._vocab_size - 1)
        self.word2id = {word: i for i, (word, _) in enumerate(most_common)}
        unk_id = len(self.word2id)
        self.word2id[self.unk_token] = unk_id
        self.id2word = {i: w for w, i in self.word2id.items()}

    def _tokenize(self, text):
        # Regex
        # \w+ : matches 1 or more word characters
        # [^\w\s] : any single character that is not a word or whitespace (punctuation)
        return re.findall(r"\w+|[^\w\s]", text.lower(), re.UNICODE)

    def encode(self, text: str) -> list[int]:
        tokens = self._tokenize(text)
        return [self.word2id.get(tok, self.word2id[self.unk_token]) for tok in tokens]

    def decode(self, ids: list[int]) -> str:
        return " ".join([self.id2word.get(i, self.unk_token) for i in ids])

    def vocab_size(self):
        return len(self.word2id)
    
    @classmethod
    def from_json(cls, path: str) -> 'WPTokenizer':
        with open(path, "r") as f:
            word2id: dict[str, int] = json.load(f)
        return WPTokenizer(len(word2id), word2id=word2id)
        
    

class GPTTokenizer:

    def __init__(self):
        self.model: AutoTokenizer = AutoTokenizer.from_pretrained("gpt2")

    def encode(self, text: str) -> list[int]:
        return self.model.encode(text, add_special_tokens=False)
    
    def decode(self, tokens: list[int]) -> str:
        return self.model.decode(tokens)
    
    def vocab_size(self) -> int:
        return self.model.vocab_size
    

class SPTokenizer:

    def __init__(self, model_path: str):
        self.sentence_piece = spm.SentencePieceProcessor()
        self.sentence_piece.load(model_path)

    def encode(self, text: str) -> list[int]:
        return self.sentence_piece.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.sentence_piece.decode(tokens)
    
    def vocab_size(self) -> int:
        return self.sentence_piece.get_piece_size()