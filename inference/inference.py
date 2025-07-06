# === inference/inference.py ===
from transformers import TextStreamer

def run_inference(model, tokenizer, prompt):
    messages = [{
        "role": "user",
        "content": [{"type": "text", "text": prompt}]
    }]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        tokenize=True,
        return_dict=True,
    ).to("cuda")
    _ = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=1.0, top_p=0.95, top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )