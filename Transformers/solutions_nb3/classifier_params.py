VOCAB_SIZE = tokenizer.vocab_size
MAX_LENGTH = tokenizer.model_max_length
HIDDEN_DIM = 256
NUM_HEADS = 8
NUM_CLASSES = tokenized_dataset["train"].features["label"].num_classes

print(f"VOCAB_SIZE: {VOCAB_SIZE}")
print(f"MAX_LENGTH: {MAX_LENGTH}")
print(f"HIDDEN_DIM: {HIDDEN_DIM}")
print(f"NUM_HEADS: {NUM_HEADS}")
print(f"NUM_CLASSES: {NUM_CLASSES}")