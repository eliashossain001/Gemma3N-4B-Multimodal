# === model/save_model.py ===
def save_model(model, tokenizer, path="gemma-3n"):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)