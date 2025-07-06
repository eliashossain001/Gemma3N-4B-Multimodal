# === utils/tokenizer_setup.py ===
from unsloth.chat_templates import get_chat_template

def setup_tokenizer(tokenizer):
    return get_chat_template(tokenizer, chat_template="gemma-3")

# === main.py ===
from model.load_model import load_model_and_tokenizer
from data.preprocess import load_and_format_dataset, formatting_prompts_func
from configs.sft_config import get_sft_config
from train_module.train import train_model
from model.save_model import save_model
from utils.memory_utils import show_memory_stats
from utils.tokenizer_setup import setup_tokenizer

model, tokenizer = load_model_and_tokenizer()
tokenizer = setup_tokenizer(tokenizer)
dataset = load_and_format_dataset()
dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)
sft_config = get_sft_config()
trainer = train_model(model, tokenizer, dataset, sft_config)

show_memory_stats()
trainer_stats = trainer.train()
show_memory_stats()
save_model(model, tokenizer)
print("Training complete.")