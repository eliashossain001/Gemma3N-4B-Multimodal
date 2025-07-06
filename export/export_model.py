# === export/export_model.py ===
def export_lora(model, tokenizer, output_path="gemma-3n-lora"):
    model.save_lora(output_path)

def export_merged(model, tokenizer, output_path="gemma-3N-finetune"):
    model.save_pretrained_merged(output_path, tokenizer)

def export_gguf(model, output_path="gemma-3N-finetune", quant="Q8_0"):
    model.save_pretrained_gguf(output_path, quantization_type=quant)