# === inference/multimodal_demo.py ===
from transformers import TextStreamer

def multimodal_demo(model, tokenizer):
    sloth_link = "https://files.worldwildlife.org/wwfcmsprod/images/Sloth_Sitting_iStock_3_12_2014/story_full_width/8l7pbjmj29_iStock_000011145477Large_mini__1_.jpg"
    audio_file = "audio.mp3"

    messages = [{
        "role": "user",
        "content": [
            {"type": "audio", "audio": audio_file},
            {"type": "image", "image": sloth_link},
            {"type": "text",  "text": "What is this audio and image about? How are they related?"}
        ]
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
        max_new_tokens=256,
        temperature=1.0, top_p=0.95, top_k=64,
        streamer=TextStreamer(tokenizer, skip_prompt=True),
    )