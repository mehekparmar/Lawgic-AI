"""
Standalone test script for the fine-tuned legal query rewriter.
Run with: python test_rewriter.py
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from peft import PeftModel

ADAPTER_DIR = "models/legal_query_rewriter_lora"
BASE_MODEL = "google/flan-t5-small"
PROMPT_PREFIX = "Rewrite this legal question using precise legal terminology: "

device = torch.device("cpu")  # force CPU to avoid MPS device-mismatch issues

print("Loading base model + adapter...")
tokenizer = T5Tokenizer.from_pretrained(ADAPTER_DIR)
base_model = T5ForConditionalGeneration.from_pretrained(BASE_MODEL).to(device)
base_model.config.tie_word_embeddings = False
model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(device)
model.eval()

def rewrite(query: str) -> str:
    inputs = tokenizer(PROMPT_PREFIX + query, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

test_queries = [
    "the seller won't take back a broken item",
    "can I get out of jail if this is my first offense",
    "my husband wants a divorce but I didn't do anything wrong",
    "the shop lied about what the product could do",
]

print("\n--- Rewrite test ---")
for q in test_queries:
    print(f"IN:  {q}")
    print(f"OUT: {rewrite(q)}\n")
