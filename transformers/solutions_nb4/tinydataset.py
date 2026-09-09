tiny_dataset = {
    "train": raw_dataset["train"].select(range(500)),  # Keep first 5,000 examples
    "validation": raw_dataset["validation"].select(range(100)),  # Keep first 1_000
    "test": raw_dataset["test"].select(range(100))  # Keep first 1_000
}

tiny_dataset = DatasetDict(tiny_dataset)
tiny_dataset