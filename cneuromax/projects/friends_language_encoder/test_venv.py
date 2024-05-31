import glob
import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

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
from transformers.tokenization_utils_base import BatchEncoding

from cneuromax.fitting.deeplearning.litmodule.base import (
    BaseLitModuleConfig,
)
from cneuromax.projects.friends_language_encoder.embedding import (
    DataConfigBase,
    PrepareFinetunedTokenizedBatches,
    get_layer_embeddding,
    prepare_embeddings,
)
from cneuromax.projects.friends_language_encoder.litmodule_peft import (
    FriendsFinetuningModel,
)
from cneuromax.projects.friends_language_encoder.utils import (
    list_episodes,
    list_seasons,
    preprocess_words,
    set_output,
    split_episodes,
)

from .utils import group_texts_hf


@dataclass
class DataConfig(DataConfigBase):
    """."""

    bold_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/fmri_data/"
    )
    stimuli_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/gpt2"
    )
    output_dir: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/ridge_regression"
    )
    tsv_path: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/"
    )
    fmri_file: str = (
        "sub-03_task-friends_space-MNI152NLin2009cAsym_atlas-MIST_desc-444_timeseries.h5"
    )
    word_tsv_path: str = (
        "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli/word_alignment/"
    )
    target_layer: int = 13
    atlas: str = "MIST"
    parcel: str = "444"
    subject_id: str = "sub-03"
    n_split: int = 7
    random_state: int = 42
    test_season = "s03"  # season allocated for test
    TR_delay = (
        5  # "How far back in time (in TRs) does the input window start "
    )
    # "in relation to the TR it predicts. E.g., back = 5 means that input "
    # "features are sampled starting 5 TRs before the target BOLD TR onset",
    duration = 3
    # "Duration of input time window (in TRs) to predict a BOLD TR. "
    # "E.g., input_duration = 3 means that input is sampled over 3 TRs "
    # "to predict a target BOLD TR.",

data_config = DataConfig()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizerBase,
    get_constant_schedule,
)

from cneuromax.projects.friends_language_encoder.activation import (
    AttachLayerHook,
)
from cneuromax.projects.friends_language_encoder.utils import (
    list_episodes,
    list_seasons,
    preprocess_words,
    set_output,
)

seasons = ["s01", "s02", "s03", "s04", "s05", "s06"]



tokenizer = AutoTokenizer.from_pretrained(data_config.base_model_name)
for season in seasons[:1]:
    print(season)
    comp_args, outfile_name = set_output(
        season=season,
        output_dir=data_config.embedding_dir,
    )
    episode_list = list_episodes(
        idir=data_config.word_tsv_path,
        season=season,
        outfile=outfile_name,
    )
    print(episode_list)

    for episode in episode_list[:1]:
        print(episode)

        model_config = BaseLitModuleConfig()
        peft_config = LoraConfig(
            task_type=data_config.task_type,
            inference_mode=data_config.inference_mode,
            r=data_config.r,
            lora_alpha=data_config.lora_alpha,
            lora_dropout=data_config.lora_dropout,
            fan_in_fan_out=data_config.fan_in_fan_out,
        )


        nnmodule = AutoModelForCausalLM.from_pretrained(
            data_config.base_model_name,
        )

        ckp_path: Path = (
            Path(data_config.model_dir)
            / data_config.base_model_name
            / f"{data_config.sweep}"
            / "lightning"
            / "last.ckpt"
        )

        model = FriendsFinetuningModel.load_from_checkpoint(
            ckp_path,
            config=model_config,
            peft_config=peft_config,
            nnmodule=nnmodule,
            optimizer=partial(Adam),
            scheduler=partial(get_constant_schedule),
        )
        # for name, module in model.named_children():
        #     print(f"{name}: {type(module)}")

        # for name, module in model.named_modules():
        #     print(f"{name}: {type(module)}")

        # for name, module in model.
        #     print(f"{name}: {type(module)}")

        # my_activations = AttachLayerHook(model, layer_name="h.11.attn" )

        # all_activations = my_activations.get_activations()
        # print(all_activations)
        # text = PrepareFinetunedTokenizedBatches(
        #     tokenizer=tokenizer,
        #     tsv_path=data_config.word_tsv_path,
        #     season=season,
        #     episode=episode,
        #     context_size=data_config.context_size,
        #     connection_character="Ġ",
        # )
        # print("text class is ready.")

        # hf_grouped_and_tokenized_dataset,hf_tokenized_dataset_tmp= text.tokenize_batch()
        # print("mapping is done.")

        # input_ids, indexes, tokens = text.get_text_batches()

            # embeddings = ExtractEmbedding(
            #     indexes=indexes,
            #     mapping=mapping,
            #     input_ids=input_ids,
            #     model=model,
            #     data_config=data_config,
            # )
