from unsloth import FastModel

def load_model_and_tokenizer(model_name="unsloth/gemma-3n-E4B-it", max_seq_length=1024):
    # Step 1: Load base model (quantized)
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )

    # Step 2: Attach LoRA after loading
    model = FastModel.get_peft_model(
        model,
        r=64,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"]
    )

    return model, tokenizer
