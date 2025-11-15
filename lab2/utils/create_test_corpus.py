from datasets import load_from_disk


if __name__ == '__main__':
    ds = load_from_disk("data/tinystories")
    train = ds["train"]

    with open("data/corpus.txt", "w", encoding="utf-8") as f:
        cnt = 0
        for text in train["text"]:
            cnt += 1
            clean_text = text.replace("\n", " ").strip()
            f.write(clean_text + "\n")

            if cnt % 100_000 == 0:
                print(cnt)