"""
LoRA fine-tuning for legal query rewriting.
Trains a small T5 model to rewrite casual legal questions into precise terminology.
Run with: python finetune_lora.py
"""

import json
from pathlib import Path
from transformers import (
    T5Tokenizer, T5ForConditionalGeneration,
    Seq2SeqTrainer, Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset

MODEL_NAME = "google/flan-t5-small"
DATA_PATH = Path("data") / "rewrite_training_data.json"
OUTPUT_DIR = Path("models") / "legal_query_rewriter_lora"

print("Loading base model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
base_model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
base_model.config.tie_word_embeddings = False

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q", "v"],  # standard for T5 attention
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()

print("Loading training data...")
with open(DATA_PATH) as f:
    raw_data = json.load(f)
print(f"Loaded {len(raw_data)} training pairs")

PROMPT_PREFIX = "Rewrite this legal question using precise legal terminology: "

def preprocess(example):
    inputs = tokenizer(
        PROMPT_PREFIX + example["input"], max_length=64, truncation=True, padding="max_length"
    )
    labels = tokenizer(
        example["output"], max_length=64, truncation=True, padding="max_length"
    )
    # Critical: replace pad-token ids with -100 so the loss ignores padding
    # positions entirely, instead of training the model to predict padding.
    label_ids = labels["input_ids"]
    label_ids = [tok if tok != tokenizer.pad_token_id else -100 for tok in label_ids]
    inputs["labels"] = label_ids
    return inputs

dataset = Dataset.from_list(raw_data).map(preprocess, remove_columns=["input", "output"])

training_args = Seq2SeqTrainingArguments(
    output_dir=str(OUTPUT_DIR),
    per_device_train_batch_size=4,
    num_train_epochs=15,
    learning_rate=3e-4,
    logging_steps=5,
    save_strategy="no",
    report_to=[],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

print("Training...")
trainer.train()

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
print(f"Saved LoRA adapter to {OUTPUT_DIR}")
