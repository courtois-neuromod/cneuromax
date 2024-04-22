from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft.tuners.lora.config import LoraConfig
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    get_constant_schedule,
)

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModuleConfig,
)

# perplexity
from cneuromax.projects.friends_language_encoder.activation import (
    AttachLayerHook,
)
from cneuromax.projects.friends_language_encoder.litmodule_peft import (
    FriendsFinetuningModel,
)


@dataclass
class DataConfig:
    """."""

    data_dir: str = "./data/friends_language_encoder/"
    base_model_name: str = "gpt2"


data_config = DataConfig()

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
    data_config.base_model_name,
)
sweeps = [
    "overrides#lr~0.0001161#wd~0.001303",
    "overrides#lr~1.836e-05#wd~0.04784",
    "overrides#lr~1.28e-05#wd~0.03811",
    "overrides#lr~2.927e-05#wd~0.006349",
    "overrides#lr~3.083e-05#wd~0.0287",
    "overrides#lr~6.892e-05#wd~0.01672",
]

for sweep in tqdm(sweeps[:1]):
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


# def extract_activations(
#     data_config: DataConfig,
# ) -> None:
#     """."""

#     model_config = BaseLitModuleConfig()
#     peft_config = LoraConfig(
#         task_type="CAUSAL_LM",
#         inference_mode=False,
#         r=8,
#         lora_alpha=32,
#         lora_dropout=0.1,
#         fan_in_fan_out=True,
#     )
#     # sweeps = [
#     #     "overrides#lr~0.0001161#wd~0.001303",
#     #     "overrides#lr~1.836e-05#wd~0.04784",
#     #     "overrides#lr~1.28e-05#wd~0.03811",
#     #     "overrides#lr~2.927e-05#wd~0.006349",
#     #     "overrides#lr~3.083e-05#wd~0.0287",
#     #     "overrides#lr~6.892e-05#wd~0.01672",
#     # ]
#     sweeps = [
#         "overrides#lr~0.0001161#wd~0.001303",
#     ]
#     print(
#         data_config.base_model_name,
#     )

#     nnmodule = AutoModelForCausalLM.from_pretrained(
#         data_config.base_model_name,
#     )
#     print(nnmodule)
#     # for sweep in tqdm(sweeps):
#     #     print(sweep)
#     #     ckp_path: Path = (
#     #         Path(data_config.data_dir)
#     #         / data_config.base_model_name
#     #         / f"{sweep}"
#     #         / "lightning"
#     #         / "last.ckpt"
#     #     )

#     #     finetuned_model = FriendsFinetuningModel.load_from_checkpoint(
#     #         ckp_path,
#     #         config=model_config,
#     #         peft_config=peft_config,
#     #         nnmodule=nnmodule,
#     #         optimizer=partial(Adam),
#     #         scheduler=partial(get_constant_schedule),
#     #     )

#     #     hook = AttachLayerHook(
#     #         nnmodule=finetuned_model,
#     #         layer_name=data_config.layer_name,
#     #     )
#     #     activation = hook.get_activations
#     #     print(activation)
#     #     print(type(activation))


# # @dataclass
# # class DataConfig:
# #     """."""

# #     data_dir: str = "./data/friends_language_encoder/"
# #     base_model_name: str = "gpt2"
# #     layer_name: str = "h.11.attn.c_attn.weight"
#     # layers: list[str] = [
#     #     "transformer.h.11.ln_1.weight",
#     #     "transformer.h.11.ln_1.bias",
#     #     "transformer.h.11.attn.c_attn.weight",
#     #     "transformer.h.11.attn.c_attn.bias",
#     #     "transformer.h.11.attn.c_proj.weight",
#     #     "transformer.h.11.attn.c_proj.bias",
#     #     "transformer.h.11.ln_2.weight",
#     #     "transformer.h.11.ln_2.bias",
#     #     "transformer.h.11.mlp.c_fc.weight",
#     #     "transformer.h.11.mlp.c_fc.bias",
#     #     "transformer.h.11.mlp.c_proj.weight",
#     #     "transformer.h.11.mlp.c_proj.bias",
#     # ]


# # extract_activations(
# #     data_config=data_config,
# # )
