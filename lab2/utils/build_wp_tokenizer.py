from tokenizer import WPTokenizer
import json


if __name__ == '__main__':
    with open("data/corpus.txt", "r") as f:
        texts = f.readlines()

    tokenizer = WPTokenizer(vocab_size=50257)
    tokenizer.build_vocab(texts)

    with open("data/custom_vocab.json", "w") as f:
        json.dump(tokenizer.word2id, f)
