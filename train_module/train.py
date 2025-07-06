# === train_module/train.py ===
from trl import SFTTrainer
from unsloth.chat_templates import train_on_responses_only

def train_model(model, tokenizer, dataset, sft_config):
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        args=sft_config,
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<start_of_turn>user\n",
        response_part="<start_of_turn>model\n",
    )
    return trainer