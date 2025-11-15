from datasets import load_dataset

if __name__ == "__main__":
    ds = load_dataset("roneneldan/TinyStories")
    ds.save_to_disk("data/tinystories")