import sys
import os
# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.getcwd()))
print(f"sys.path = {sys.path}")

from tests import adapters
import time
from memory_profiler import profile

@profile
def run_train_bpe(input_path: str, vocab_size: int, special_tokens: list):
    vocab, merges = adapters.run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
    )
    return vocab, merges


if __name__ == "__main__":
    # Run the BPE training
    start_time = time.time()
    vocab, merges = run_train_bpe(
        "../data/TinyStoriesV2-GPT4-valid.txt", 
        vocab_size=10_000, 
        special_tokens=["<|endoftext|>"])
    end_time = time.time()

    print(f"Training BPE took {end_time - start_time:.2f} seconds")