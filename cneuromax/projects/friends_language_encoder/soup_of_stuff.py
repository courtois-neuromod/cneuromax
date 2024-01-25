r""".

# def split_data(stimuli_path):
#     data = pd.read_csv(
#         os.path.join(path, "segments.csv"), header=None
#     ).tolist()  # Specify header=None
#     num_samples = len(data)
#     percentage_to_leave_out = 20
#     stimuli_data = os.listdir(stimuli_path)

#     # Calculate the number of samples to leave out
#     num_samples_to_leave_out = int(percentage_to_leave_out / 100 * num_samples)

#     # Create a LeavePOut iterator
#     logo = LeavePOut(p=num_samples_to_leave_out)

#     splits = []

#     gentles = [s["onset"].values for s in stimuli_data]
#     nscans = [f.shape[0] for f in fmri_data]  # number of scnas per session

#     for train, test in logo.split(features):
#         # Compute the number of rows in each run (= the number of samples extracted from the model for each run)
#         gentles_train = [gentles[i] for i in train]
#         groups_train = get_groups(gentles_train)

#         gentles_test = [gentles[i] for i in test]
#         groups_test = get_groups(gentles_test)
#         # Preparing fMRI data

#         splits.append(
#             {
#                 "fmri_train": [fmri_data[i] for i in train],
#                 "fmri_test": [fmri_data[i] for i in test],
#                 "features_train": [features[i] for i in train],
#                 "features_test": [features[i] for i in test],
#                 "groups_train": groups_train,
#                 "nscans_train": [nscans[i] for i in train],
#                 "gentles_train": gentles_train,
#                 "groups_test": groups_test,
#                 "nscans_test": [nscans[i] for i in test],
#                 "gentles_test": gentles_test,
#             }
#         )


# # def create_data_path_list(
# #     path: str,
# #     episode: str,
#     subject: str,
#     use_bold: bool = False,
# ):
#

#     Load stimuli data from path.
#     Download it if not already done.

#     Args:
#         - path: str

#     Returns:
#         - data_list: list of csv
#
#     if use_bold:
#         nifti_data_dir = os.path.join(
#             path,
#             "bold",
#         )
#         data_path: list[str] = glob.glob(
#             f"{nifti_data_dir}/{subject}/ses-0*/func/"
#             f"{subject}_ses-0*_task-{episode}_space-T1w_desc-preproc_bold.nii.gz",
#         )
#     else:
#         stimuli_data_dir = os.path.join(path, "stimuli", "data")
#         print(stimuli_data_dir)
#         data_path: list[str] = glob.glob(
#             f"{stimuli_data_dir}/s0*/" f"aligned_{episode}.tsv",
#         )

#     return data_path


# def load_model_and_tokenizer(pretrained_model):
#     .

#     Load a HuggingFace model and the associated tokenizer given its name.

#     Args:
#         - trained_model: str

#     Returns:
#         - model: HuggingFace model
#         - tokenizer: HuggingFace tokenizer.
#

#     if pretrained_model == "gpt2":
#         model = AutoModelForCausalLM.from_pretrained(pretrained_model)
#         tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#     else:
#         model = AutoModelForMaskedLM.from_pretrained(pretrained_model)
#         tokenizer = AutoTokenizer.from_pretrained(pretrained_model)
#     return model, tokenizer


# def extract_segment_names(path):
#     .

#     Creates the list of video segments for further processing

#     Arguments:
#       - path to the file
#     Returns:
#       - .csv file of the segment names.

#
#     segments = []

#     for i in range(1, 7):
#         data = pd.read_csv(
#             os.path.join(path, f"s{i}_wav_list.csv"), header=None
#         )  # Specify header=None
#         for _, item in data.iterrows():
#             # print(item[0])  # Access the first (and only) column in each row
#             segment_name = item[0].replace(".wav", "").replace("friends_", "")
#             segments.append(segment_name)

#     # Create a DataFrame and save it to a CSV file
#     segments_df = pd.DataFrame({"segments": segments})
#     segments_df.to_csv(os.path.join(path, "segments.csv"), index=False)


# def create_h5py(
#     data_dir: str,
#     lazy_load_path: str,
#     subject: str | None,
#     stage: str | None,
#     load_confounds_params: None = None,
#     add_confounds: bool = False,
#     mask_dir: str | None = None,
#     fwhm: int | None = None,
#     use_bold: bool = False,
# ) -> None:
#     .

#     Loads data (bold).

#     Applies mask and preprocessing.

#     Create and store as h5py file.
#
#     if use_bold:
#         with h5py.File(
#             os.path.join(lazy_load_path, subject, f"fmri_{stage}.h5"), "w"
#         ) as hf:
#             bold_data = hf.create_group("bold")
#             mask_data = nib.load(mask_dir)
#             masker = NiftiLabelsMasker(
#                 mask_data,
#                 standardize=True,
#                 detrend=True,
#                 smoothing_fwhm=fwhm,
#             )
#             fmri_files = os.listdir(os.path.join(data_dir, subject, stage))
#             index = 1
#             for file in fmri_files:
#                 nii_data = os.path.join(data_dir, subject, stage, file)
#                 masker.fit(nii_data)
#                 if add_confounds == True:
#                     confounds, _ = load_confounds(
#                         nii_data,
#                         **OmegaConf.to_container(load_confounds_params),
#                     )
#                     fmri_data = masker.transform(nii_data, confounds=confounds)
#                 else:
#                     fmri_data = masker.transform(nii_data, confounds=confounds)
#                     # print(fmri_data.shape)
#             bold_data.create_dataset(f"fmri{index}", data=fmri_data)
#         hf.close()
#     else:
#         with h5py.File(
#             os.path.join(lazy_load_path, subject, f"stimuli_{stage}.h5"),
#             "w",
#         ) as hf:
#             word_tokens = hf.create_group("tokens")
#             tokens = tokenize_words(alignment_path)
#             word_tokens.create_dataset(f"tokens{index}", data=tokens)
#             time_stamps = hf.create_group("times")
#             timing = pd.read_csv(timing_path, sep="\t", usecols=["onset"])
#             time_stamps.create_dataset(f"time_stamps{index}", data=timing)
#         hf.close()
#         hf.close()
"""  # noqa: W505, E501
