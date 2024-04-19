"""."""

from typing import Any

import numpy as np
import torch
from datasets import load_metric
from jaxtyping import Float
from torch import Tensor, nn
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
    Trainer,
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

    def get_finetune_perplexity(
        self: "ExtractPerplexity",
    ) -> Float[Tensor, ""]:
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

    def get_base_perplexity(self: "ExtractPerplexity") -> Float[Tensor, ""]:
        """Method for getting perplexity."""
        stepwise_perplexity = []
        prev_end_loc = 0
        for begin_loc in tqdm(range(0, self.seq_len, self.step)):
            end_loc = min(begin_loc + self.max_length, self.seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = self.text.input_ids[:, begin_loc:end_loc]
            labels = input_ids.clone()
            labels[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.nnmodule(input_ids, labels=labels)
                neg_log_likelihood = outputs.loss

            stepwise_perplexity.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == self.seq_len:
                break

        return torch.exp(torch.stack(stepwise_perplexity).mean())


class EvaluateBenchmark:
    """."""

    def __init__(
        self: "EvaluateBenchmark",
        actual_task: str,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        nnmodule: nn.Module,
        eval_dataset: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        metric: "str",
    ) -> None:
        """."""
        self.actual_task = actual_task
        self.predictions = predictions
        self.labels = labels
        self.model = nnmodule
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        self.metric = load_metric(metric, self.actual_task)
        self.metric.compute(
            predictions=self.predictions,
            references=self.labels,
        )

    def compute_metrics(
        self: "EvaluateBenchmark",
        eval_pred: torch.Tensor,
        task: str,
    ) -> Any:
        """."""
        logits, labels = eval_pred
        if task != "stsb":
            predictions = np.argmax(
                logits, axis=1
            )  # check if -1 is true too from here https://github.com/huggingface/transformers/blob/b109257f4fb8b1166e7c53cc5418632014ed53a5/docs/source/en/training.md?plain=1#L25
        else:
            predictions = predictions[:, 0]
        return self.metric.compute(predictions=predictions, references=labels)

    def compute_evaluation(
        self: "EvaluateBenchmark",
    ) -> Any:
        """."""
        trainer = Trainer(
            model=self.model,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )
        return trainer.evaluate()
