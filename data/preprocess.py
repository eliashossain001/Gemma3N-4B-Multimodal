# === data/preprocess.py ===
from datasets import load_dataset
from unsloth.chat_templates import standardize_data_formats

def load_and_format_dataset():
    dataset = load_dataset("mlabonne/FineTome-100k", split="train[:3000]")
    dataset = standardize_data_formats(dataset)
    return dataset

def formatting_prompts_func(examples, tokenizer):
    texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False).removeprefix('<bos>') for convo in examples["conversations"]]
    return {"text": texts}