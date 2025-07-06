from model.load_model import load_model_and_tokenizer
from data.preprocess import load_and_format_dataset, formatting_prompts_func
from configs.sft_config import get_sft_config
from train_module.train import train_model
from model.save_model import save_model
from utils.memory_utils import show_memory_stats
from utils.tokenizer_setup import setup_tokenizer

# Load model and tokenizer
model, tokenizer = load_model_and_tokenizer()

# Setup tokenizer with chat template
tokenizer = setup_tokenizer(tokenizer)

# Load and preprocess dataset
dataset = load_and_format_dataset()
dataset = dataset.map(lambda x: formatting_prompts_func(x, tokenizer), batched=True)

# Get SFT training configuration
sft_config = get_sft_config()

# Train model
trainer = train_model(model, tokenizer, dataset, sft_config)

# Show memory usage before and after training
show_memory_stats()
trainer_stats = trainer.train()
show_memory_stats()

# Save model and tokenizer
save_model(model, tokenizer)
print("✅ Training complete.")
