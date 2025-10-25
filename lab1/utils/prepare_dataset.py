import json
import os
import random
from speakleash import Speakleash

def prepare_dataset(
    sl: Speakleash,
    dataset_name: str,
    output_dir: str,
    char_limit: int,
    num_shards: int = 1,
):
    
    if num_shards < 1:
        raise ValueError(f"The number of output shards should be at least 1, given: {num_shards}")

    os.makedirs(output_dir, exist_ok=True)
    dataset = sl.get(dataset_name)

    shard_files = []

    for i in range(num_shards):
        path = os.path.join(output_dir, f"docs_{i+1:05d}.jsonl")
        file_handle = open(path, "w", encoding="utf-8")
        shard_files.append(file_handle)

    total_chars = 0

    for doc in dataset.data:
        if total_chars >= char_limit:
            break

        if random.uniform(0, 1) > 0.2: # Take every one fifth doc
            continue

        record = {"text": doc}
        shard_idx = random.randint(0, num_shards - 1)
        shard_files[shard_idx].write(json.dumps(record, ensure_ascii=False) + "\n")
        total_chars += len(doc)

    for f in shard_files:
        f.close()

    print(f"Finished writing {num_shards} shards with total ~{total_chars:,} characters.")



def main():
    sl = Speakleash("../speakleash_data")

    prepare_dataset(sl, "news_5_lifestyle_corpus", "../data/train/", 50_000_000, 10)
    prepare_dataset(sl, "plwiki", "../data/test/", 1_000_000)


if __name__ == '__main__':
    main()
    