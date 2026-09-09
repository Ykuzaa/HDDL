print(f"Name of the tokenizer: {tokenizer.__class__}")
print(f"Size of the vocabulary: {tokenizer.vocab_size}")
print(f"Maximum model input length: {tokenizer.model_max_length}")
print(f"Special tokens: {tokenizer.special_tokens_map}")

for key, value in tokenizer.special_tokens_map.items():
    print(f"{key}: {value}; token_id: {tokenizer.convert_tokens_to_ids(value)}")