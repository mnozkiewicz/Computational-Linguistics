from datasets import load_dataset

if __name__ == "__main__":
    # ds = load_dataset("roneneldan/TinyStories")
    # ds.save_to_disk("data/tinystories")

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    ds.save_to_disk("data/wiki")