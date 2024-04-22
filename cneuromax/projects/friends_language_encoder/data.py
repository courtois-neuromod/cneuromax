"""."""

import glob
import logging
import os
import string
from collections import defaultdict
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from nilearn import image, input_data, masking
from src.utils import read_yaml, save_yaml
from tqdm import tqdm
from transformers import (
    PreTrainedTokenizerBase,
)

from cneuromax.projects.friends_language_encoder.utils import (
    read_tsv,
    read_yaml,
)


class ProcessText:
    """."""

    def __init__(
        self: "ProcessText",
        tokenizer: PreTrainedTokenizerBase,
        tsv_path: str,
        season: int,
        episode: str,
        context_size: int,
        connection_character: str = "Ġ",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_length = tokenizer.model_max_length
        self.tsv_path = os.path.join(
            tsv_path,
            f"s{season}",
            f"friends_{episode}.tsv",
        )
        self.season = season
        self.episode = episode
        self.connection_character = connection_character
        self.context_size = context_size

    def preprocess_words(self: "ProcessText") -> str:
        """Un-punctuate, lower and combine the words like a text.

        Args:
            - tsv_path: path to the episode file
        Returns:
            - list of concatanated words
        """
        data = read_tsv(self.tsv_path)
        stimuli_data = data["word"].apply(
            lambda x: x.translate(
                str.maketrans("", "", string.punctuation),
            ).lower(),
        )

        return " ".join(stimuli_data)

    def tokenize_and_match(
        self: "ProcessText",
    ) -> defaultdict[int, list[int]]:
        """Tokenize and map sub-tokens with the words.

        Returns:
            - mapping: A dictionary {int: list(int)} mapping of
            untokenized words with the subtokens.
        """
        eos_token = "<|endoftext|>"
        untokenized_text = self.preprocess_words()
        tokenized_text = self.tokenizer.tokenize(untokenized_text)
        mapping = defaultdict(list)
        untokenized_text_index = 0
        tokenized_text_index = 0
        while untokenized_text_index < len(
            untokenized_text,
        ) and tokenized_text_index < len(tokenized_text):
            while (
                tokenized_text_index + 1 < len(tokenized_text)
                and (
                    not tokenized_text[tokenized_text_index + 1].startswith(
                        self.connection_character,
                    )
                )
                and tokenized_text[tokenized_text_index + 1] != eos_token
            ):
                mapping[untokenized_text_index].append(tokenized_text_index)
                tokenized_text_index += 1
            mapping[untokenized_text_index].append(untokenized_text_index)
            untokenized_text_index += 1
            tokenized_text_index += 1
        return mapping

    def pad_to_max_length(
        self: "ProcessText",
        sequence: list[Any],
        padding_value: Any = 220,
        end_token: Any = 50256,
    ) -> Any:
        """Pag sequence to reach max_seq_length.

        Args:
            - sequence: text chunked in context size
            - padding_value: int (default 220)
            - end_token: int (default 50256)

        Returns:
            - result: padded texts as list of int
        """
        sequence = sequence[: self.max_seq_length]
        padding_required = self.max_seq_length - len(sequence)
        result = sequence + [padding_value, end_token] * (
            padding_required // 2
        )
        if len(result) == self.max_seq_length:
            return result
        else:
            return result + [padding_value]

    def create_examples(
        self: "ProcessText",
        sequence: list[Any],
        padding_value: Any = 220,
        start_token: Any = 50256,
        end_token: Any = 50256,
    ) -> Any:
        """Returns list of InputExample objects.

        Args:
            - sequence: text chunked in context size
            - padding_value: int (default 220)
            - end_token: int (default 50256)
            - end_token: int (default 50256)

        Returns:
            - result: padded texts as list of int
            - start_token: int (default 50256)
            - end_token: int (default 50256)
        """
        new_sequence = [start_token] + sequence + [padding_value, end_token]
        return self.pad_to_max_length(new_sequence)

    def get_text_chunks(
        self: "ProcessText",
        padding_value: Any = 220,
        start_token: str = "<|endoftext|>",
        end_token: str = "<|endoftext|>",
    ) -> Any:
        """."""
        stimuli = self.preprocess_words()

        if self.context_size is None:
            self.max_seq_length = self.max_seq_length
        else:
            self.max_seq_length = (
                self.context_size + 5
            )  # count for special tokens
        try:
            data = self.tokenizer.encode(stimuli).ids
            text = self.tokenizer.encode(stimuli).tokens
        except:
            data = self.tokenizer.encode(stimuli)
            text = self.tokenizer.tokenize(stimuli)

        if self.context_size == 0:
            examples = [
                self.create_examples(data[i : i + 2])
                for i, _ in enumerate(data)
            ]
            tokens = [
                self.create_examples(
                    text[i : i + 2],
                    start_token=start_token,
                    end_token=end_token,
                    padding_value=self.connection_character,
                )
                for i, _ in enumerate(text)
            ]
        else:
            examples = [
                self.create_examples(data[i : i + self.context_size + 2])
                for i, _ in enumerate(data[: -self.context_size])
            ]
            tokens = [
                self.create_examples(
                    text[i : i + self.context_size + 2],
                    padding_value=padding_value,
                    start_token=start_token,
                    end_token=end_token,
                )
                for i, _ in enumerate(text[: -self.context_size])
            ]

        features = [
            torch.FloatTensor(example).unsqueeze(0).to(torch.int64)
            for example in examples
        ]
        input_ids = torch.cat(features, dim=0)
        indexes = [(1, self.context_size + 2)] + [
            (self.context_size + 1, self.context_size + 2)
            for i in range(1, len(input_ids))
        ]
        del examples
        del features
        return input_ids, indexes, tokens


class ProcessFMRI:
    """."""

    def __init__(
        self: "ProcessFMRI",
        in_path: str,
        out_path: str,
    ) -> None:
        self.in_path = in_path
        self.out_path = out_path

    def fetch_masker(
        self: "ProcessFMRI", fmri_data: Any, **kwargs: Any
    ) -> Any:
        """Fetch or compute if needed a masker from fmri_data.

        Arguments:
            - masker_path: str
            - fmri_data: list of NifitImages/str
        """
        if os.path.exists(self.out_path + ".nii.gz") and os.path.exists(
            self.out_path + ".yml"
        ):
            masker = self.load_masker(self.out_path, **kwargs)
        else:
            masks = [masking.compute_epi_mask(f) for f in fmri_data]
            mask = image.math_img(
                "img>0.5",
                img=image.mean_img(masks),
            )  # take the average mask and threshold at 0.5
            masker = input_data.NiftiMasker(mask, **kwargs)
            masker.fit()
            self.save_masker(masker)
        return masker

    def load_masker(
        self: "ProcessFMRI",
        resample_to_img: Any,
        intersect_with_img: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Given a path without the extension.

        load the associated yaml and Nifti files
        to compute the associated masker.

        Arguments:
            - path: str
            - resample_to_img_: Nifti image (optional)
            - intersect_with_img: bool (optional)
            - kwargs: dict
        """
        params = read_yaml(self.in_path + ".yml")
        mask_img = nib.load(self.in_path + ".nii.gz")

        if resample_to_img is not None:
            mask_img = image.resample_to_img(
                mask_img,
                resample_to_img,
                interpolation="nearest",
            )
            if intersect_with_img:
                mask_img = self.intersect_binary(mask_img, resample_to_img)
        masker = input_data.NiftiMasker(mask_img)
        masker.set_params(**params)
        if kwargs:
            masker.set_params(**kwargs)
        masker.fit()
        return masker

    def save_masker(self: "ProcessFMRI", masker: Any) -> None:
        """Save the yaml file and image associated with a masker."""
        params = masker.get_params()
        params = {
            key: params[key]
            for key in [
                "detrend",
                "dtype",
                "high_pass",
                "low_pass",
                "mask_strategy",
                "memory_level",
                "smoothing_fwhm",
                "standardize",
                "t_r",
                "verbose",
            ]
        }
        nib.save(masker.mask_img_, self.out_path + ".nii.gz")
        save_yaml(params, self.out_path + ".yml")

    def intersect_binary(self: "ProcessFMRI", img1: Any, img2: Any) -> Any:
        """Compute the intersection of two binary nifti images.

        Arguments:
            - img1: NifitImage
            - img2: NifitImage
        Returns:
            - intersection: NifitImage
        """
        intersection = image.math_img(
            "img==2", img=image.math_img("img1+img2", img1=img1, img2=img2)
        )
        return intersection

    def preprocess_fmri_data(
        self: "ProcessFMRI",
        fmri_data: Any,
        masker: Any,
        add_noise_to_constant: bool = True,
    ) -> Any:
        """Load fMRI data and mask it with a given masker.

        Preprocess it to avoid NaN value when using Pearson
        Correlation coefficients in the following analysis.
        Returns numpy arrays, by extracting cortex activations
        using a NifitMasker.

        Args:
            - fmri_data: list of NifitImages/str
            - masker:  NiftiMasker object
            - add_noise_to_constant: Boolean
        Returns:
            - fmri_data: list of np.Array
        """
        fmri_data = [masker.transform(f) for f in fmri_data]
        # voxels with activation at zero at each time step generate
        # a nan-value pearson correlation => we add a small variation
        # to the first element
        if add_noise_to_constant:
            for index in range(len(fmri_data)):
                zero = np.zeros(fmri_data[index].shape[0])
                new = zero.copy()
                new[0] += np.random.random() / 1000
                fmri_data[index] = np.apply_along_axis(
                    lambda x: x if not np.array_equal(x, zero) else new,
                    0,
                    fmri_data[index],
                )
        return fmri_data
