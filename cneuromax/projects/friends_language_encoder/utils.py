"""``project`` utilities."""
from typing import Any

from datasets.formatting.formatting import LazyBatch


def group_texts_hf(examples: Any, block_size: Any) -> Any:  # noqa:ANN401,D103
    # Concatenate all texts.
    concatenated_examples: Any = {
        k: sum(examples[k], []) for k in examples.keys()  # noqa: SIM118
    }
    total_length = len(
        concatenated_examples[list(examples.keys())[0]],  # noqa: RUF015
    )
    # We drop the small remainder, we could add padding if the model
    # supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def group_texts(
    text: LazyBatch,
    chunk_size: int,
) -> dict[str, list[str]]:
    """.

    Concatanates and chunks the data into custom chunk size
    (mostly max_seq_length)

    Args:
      text: text data in huggingface dataset format
      chunk_size: custom size of the word sequences

    """
    # Concatenate all texts
    concatenated_text: Any = {key: sum(text[key], []) for key in text}
    # Compute length of concatenated texts
    total_length = len(
        concatenated_text[list(text.keys())[0]],  # noqa: RUF015
    )
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [
            t[i : i + chunk_size]
            for i in range(
                0,
                total_length - chunk_size + 1,
                1,
            )  # 1 is overlap
        ]
        for k, t in concatenated_text.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result
