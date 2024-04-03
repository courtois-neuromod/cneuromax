"""."""

# from torchmetrics.text import Perplexity
import torch
from torch import Tensor, nn
from jaxtyping import Float, Int


class ExtractPerplexity:
    """."""

    def __init__(
        self: "ExtractPerplexity",
        nnmodule: nn.Module,
        input_ids: Tensor,
        step: int,
    ) -> None:
        """."""
        self.nnmodule = nnmodule
        self.input_ids = input_ids
        self.seq_len = input_ids.size(1)
        self.max_length = self.nnmodule.config.n_positions
        self.step = step

    def get_perplexity(self: "ExtractPerplexity") -> Float[Tensor, ""]:
        """."""
        # TODO:
        # - adaptate to pytorch ligthning perplexity


        stepwise_perplexity = []
        prev_end_loc = 0
        for begin_loc in range(0, self.seq_len, self.step):
            end_loc = min(begin_loc + self.max_length, self.seq_len)
            trg_len = end_loc - prev_end_loc
            input_ids = self.input_ids[:, begin_loc:end_loc]
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.nnmodule(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            stepwise_perplexity.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == self.seq_len:
                break

        return torch.exp(torch.stack(stepwise_perplexity).mean())
