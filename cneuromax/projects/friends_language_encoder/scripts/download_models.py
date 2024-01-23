from transformers import AutoConfig, AutoTokenizer

# model_path =
#     '/home/isil/ComplexBrains/cneuromax/data/friends_language_encoder/model/'

# roberta-base
# bert-base-uncased
# distilbert-base-uncased
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
config = AutoConfig.from_pretrained("distilbert-base-uncased")

tokenizer.save_pretrained(
    "data/friends_language_encoder/models/distilbert-base-uncased",
)
config.save_pretrained(
    "data/friends_language_encoder/models/distilbert-base-uncased",
)
tokenizer_path = "data/friends_language_encoder/models/distilbert-base-uncased"


tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_path,
    config=AutoConfig.from_pretrained("distilbert-base-uncased"),
)

print(tokenizer.tokenize("here I am!"))
