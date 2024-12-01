from transformers import GPT2LMHeadModel, AutoTokenizer
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
import torch

from preprocess_data import preprocess_function

from rouge_score import rouge_scorer

from datasets import load_dataset

from calculate_accuracy import calculate_accuracy


tokenizer = AutoTokenizer.from_pretrained("gpt2-medium", model_max_length = 512)

model = GPT2LMHeadModel.from_pretrained('gpt2-medium', pad_token_id=tokenizer.eos_token_id)


# Load the dataset
dataset = load_dataset("cnn_dailymail", "3.0.0")


tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=["article", "highlights", "id"])

# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=1000
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./t5_finetuned_cnn_dailymail")


# Define data collator
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=1,
    weight_decay=0.01,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=500,
    save_steps=1000
)

# Define trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Train the model
trainer.train()

# Save the model
trainer.save_model("./t5_finetuned_cnn_dailymail")



model_path = "./t5_finetuned_cnn_dailymail" # model path
tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

score = calculate_accuracy(model)

print(score)

