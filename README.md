# Computational-Linguistics

Repository contains solutions for the tasks from Computational-Linguistics (smarter-sounding name for Natural Language Processing) course at AGH.
The laboratory descritpions are available [here](https://github.com/apohllo/computational-linguistics).

## Course topics

| Lab | Topics | Summary of Work |
|-------|--------|----------|
| **Lab 1** | RNNs and Transformers | Trained and compared models for next token prediction task. One model was based on Transformer (decoder only) architecture and the other on LSTM network. Both trained from scratch (or at least tried learning).
| **Lab 2** | Tokenizers | Trained and compared decoder-only generative models. Each one was trained using different tokenizer: SentencePiece, GPT2-tokenizer, Whitespace tokenizer (own implementation).
| **Lab 3** |  Fine-tuning | Built two sentiment classification models: a fine-tuned bert-base-uncased model and a compact transformer trained entirely from scratch. Compared training efficiency and classification performance. |
| **Lab 4** | Optimazation | Compared different methods for optimazing training process, including: flash-attention, windowed-attention, mixed-precision training and gradient checkpointing|
