from dataclasses import dataclass
from functools import partial
from pathlib import Path

from peft.tuners.lora.config import LoraConfig
from torch.optim import Adam
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_constant_schedule,
)

from cneuromax.fitting.deeplearning.litmodule import (
    BaseLitModuleConfig,
)
from cneuromax.projects.friends_language_encoder.datamodule import (
    create_dataset,
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

    # data_dir: str = "./data_/friends_language_encoder/"
    data_dir: str = (
        "./scratch/ibilgin/Dropbox/cneuromax/data_/friends_language_encoder/"
    )
    base_model_name: str = "gpt2_xl"
    swap = "overrides#lr~0.0004793#wd~0.008964"
    ckp_path: Path = (
        Path(data_dir)
        / base_model_name
        / f"{swap}"
        / "lightning"
        / "last.ckpt"
    )


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
# print(data_config.ckp_path)

nnmodule = AutoModelForCausalLM.from_pretrained("gpt2-xl")

model1 = FriendsFinetuningModel(
    config=model_config,
    peft_config=peft_config,
    nnmodule=nnmodule,
    optimizer=partial(Adam),
    scheduler=partial(get_constant_schedule),
)
print(model1)


model2 = FriendsFinetuningModel.load_from_checkpoint(
    data_config.ckp_path,
    config=model_config,
    peft_config=peft_config,
    nnmodule=nnmodule,
    optimizer=partial(Adam),
    scheduler=partial(get_constant_schedule),
)

print(model2)

# model = FriendsFinetuningModel(
#     config=model_config,
#     peft_config=peft_config,
#     nnmodule=nnmodule,
#     optimizer=partial(Adam),
#     scheduler=partial(get_constant_schedule),
# )

# tokenizer = AutoTokenizer.from_pretrained(data_config.base_model_name)


# test_data = create_dataset(
#     csv_file_path=Path(data_config.data_dir) / "test.csv",
#     tokenizer=tokenizer,
# )


# # perplexity = ExtractPerplexity(
# #     nnmodule=nnmodule,
# #     input_ids =
# #     step =
# # )
