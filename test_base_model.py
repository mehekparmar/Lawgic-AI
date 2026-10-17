"""Diagnostic: test the BASE model alone, no LoRA adapter, no fine-tuning."""
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

device = torch.device("cpu")
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small").to(device)
model.eval()

test_prompts = [
    "translate English to German: How are you?",
    "Answer the question: What is the capital of France?",
]

for p in test_prompts:
    inputs = tokenizer(p, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=64)
    print(f"IN:  {p}")
    print(f"OUT: {tokenizer.decode(out[0], skip_special_tokens=True)}\n")
