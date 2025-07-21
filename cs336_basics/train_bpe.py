import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Tuple

@dataclass
class BpeTrainingResult:
    """
    A container for the results of BPE tokenizer training.

    Attributes:
        vocab (Dict[int, bytes]):
            The tokenizer vocabulary, mapping token IDs to bytes.
        merges (List[Tuple[bytes, bytes]]):
            The ordered list of BPE merges created during training.
    """
    vocab: Dict[int, bytes]
    merges: List[Tuple[bytes, bytes]]

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str]
) -> BpeTrainingResult:
    pass