def preprocess_function(examples, tokenizer):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=150, truncation=True, padding="max_length")
    labels["input_ids"] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_seq]
        for labels_seq in labels["input_ids"]
    ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
