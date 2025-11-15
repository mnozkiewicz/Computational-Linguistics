import sentencepiece as spm

if __name__ == '__main__':
    spm.SentencePieceTrainer.train(
        input='data/corpus.txt',
        model_prefix='tokenizer_bpe',
        vocab_size=50257,
        model_type='bpe',
        character_coverage=1.0,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3
    )