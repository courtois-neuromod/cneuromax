from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from datasets.arrow_dataset import Dataset as HuggingfaceDataset
from peft.tuners.lora.config import LoraConfig
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_constant_schedule,
)

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModuleConfig,
)
from cneuromax.projects.friends_language_encoder.litmodule_peft import (
    FriendsFinetuningModel,
)

# perplexity
from cneuromax.projects.friends_language_encoder.performance import (
    ExtractPerplexity,
)


@dataclass
class DataConfig:
    """."""

    data_dir: str = "./data/friends_language_encoder/"
    base_model_name: str = "gpt2"


data_config = DataConfig()
tokenizer = AutoTokenizer.from_pretrained(data_config.base_model_name)
csv_file_path = Path(data_config.data_dir) / "test.csv"


def create_hf_dataset(
    csv_file_path: Path,
    tokenizer: PreTrainedTokenizerBase,
) -> HuggingfaceDataset:
    """."""
    data_df = pd.read_csv(
        csv_file_path,
        encoding="ISO-8859-1",
    )
    data_df = data_df["line"].to_frame()
    data_df = data_df.dropna()
    data_df = data_df.reset_index(drop=True)
    concatenated_string = " ".join(data_df["line"])

    return tokenizer(
        concatenated_string,
        return_tensors="pt",
    )


encodings = create_hf_dataset(
    csv_file_path,
    tokenizer=tokenizer,
)


def estimate_finetuned_model_perplexity(
    encodings: torch.Tensor,
    data_config: DataConfig,
) -> None:
    """."""

    model_config = BaseLitModuleConfig()
    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        fan_in_fan_out=True,
    )

    nnmodule = AutoModelForCausalLM.from_pretrained(
        data_config.base_model_name
    )
    sweeps = [
        "overrides#lr~0.0001161#wd~0.001303",
        "overrides#lr~1.836e-05#wd~0.04784",
        "overrides#lr~1.28e-05#wd~0.03811",
        "overrides#lr~2.927e-05#wd~0.006349",
        "overrides#lr~3.083e-05#wd~0.0287",
        "overrides#lr~6.892e-05#wd~0.01672",
    ]

    print(encodings)
    perplexity_value: dict[str, Any] = {}
    perplexity_value["sweep"] = []
    perplexity_value["perplexity"] = []

    for sweep in tqdm(sweeps):
        print(sweep)
        ckp_path: Path = (
            Path(data_config.data_dir)
            / data_config.base_model_name
            / f"{sweep}"
            / "lightning"
            / "last.ckpt"
        )

        finetuned_model = FriendsFinetuningModel.load_from_checkpoint(
            ckp_path,
            config=model_config,
            peft_config=peft_config,
            nnmodule=nnmodule,
            optimizer=partial(Adam),
            scheduler=partial(get_constant_schedule),
        )

        perplexity = ExtractPerplexity(
            nnmodule=finetuned_model,
            encoding=encodings,
            step=512,
            tokenizer=tokenizer,
        )
        perplexity_value["sweep"] = sweep
        print(sweep)
        perplexity_value["value"] = perplexity.get_finetune_perplexity()
        print(perplexity.get_finetune_perplexity())

    df = pd.DataFrame(perplexity_value)
    df.to_csv(
        "perplexity_value.csv",
        index=False,
    )  # Avoid including index column


def estimate_base_model_perplexity(
    encodings: torch.Tensor,
    data_config: DataConfig,
) -> None:
    """."""
    nnmodule = AutoModelForCausalLM.from_pretrained(
        data_config.base_model_name,
    )

    perplexity = ExtractPerplexity(
        nnmodule=nnmodule,
        encoding=encodings,
        step=512,
        tokenizer=tokenizer,
    )
    print(perplexity.get_base_perplexity())


estimate_base_model_perplexity(
    encodings=encodings,
    data_config=data_config,
)
