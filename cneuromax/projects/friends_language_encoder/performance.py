"""."""

import torch
from jaxtyping import Float
from torch import Tensor, nn
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
)
from transformers.tokenization_utils_base import BatchEncoding


class ExtractPerplexity:
    """."""

    def __init__(
        self: "ExtractPerplexity",
        nnmodule: nn.Module,
        encoding: BatchEncoding,
        tokenizer: PreTrainedTokenizerBase,
        step: int,
    ) -> None:
        """."""
        self.nnmodule = nnmodule
        self.text = encoding
        self.seq_len = len(encoding["input_ids"][0])
        self.max_length = tokenizer.model_max_length
        self.step = step

    def get_perplexity(self: "ExtractPerplexity") -> Float[Tensor, ""]:
        """Method for getting perplexity."""
        stepwise_perplexity = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, self.seq_len, self.step)):
            batch = {}
            end_loc = min(begin_loc + self.max_length, self.seq_len)
            trg_len = end_loc - prev_end_loc
            batch["input_ids"] = self.text["input_ids"][:, begin_loc:end_loc]
            batch["attention_mask"] = self.text["attention_mask"][
                :,
                begin_loc:end_loc,
            ]

            labels = batch["input_ids"].clone()
            labels[:, :-trg_len] = -100
            batch["labels"] = labels
            batch = BatchEncoding(batch)

            with torch.no_grad():
                neg_log_likelihood = self.nnmodule.step(
                    batch=batch,
                    stage="test",
                )

            stepwise_perplexity.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == self.seq_len:
                break

        return torch.exp(torch.stack(stepwise_perplexity).mean())
