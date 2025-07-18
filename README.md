# 🦙 Gemma3N_(4B)-Conversational: Multimodal Fine-Tuning using Unsloth

This project demonstrates how to fine-tune and deploy **Gemma 3N (4B)** for **multimodal conversational tasks** using the [Unsloth](https://github.com/unslothai/unsloth) fast fine-tuning framework. It supports **vision, audio, and text** inputs through a unified architecture.

---

## 📁 Project Structure

```text
<root>
├── configs/                # Configuration files
├── data/                   # Dataset preprocessing
├── model/                  # Model loading/saving
├── train_module/           # Training logic
├── inference/              # Inference scripts
├── export/                 # Export to LoRA, merged, GGUF
├── utils/                  # Utilities (tokenizer, memory)
├── main.py                 # Main training entry point
├── requirements.txt        # Project dependencies
├── .gitignore              # Ignored files
└── README.md               # Project guide
```

---

## Quick Start

### Installation
Paste the following inside a Jupyter Notebook or terminal:

```python
%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
    !pip install --no-deps unsloth
```

Also install the latest `transformers` for **Gemma 3N**:

```python
%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps git+https://github.com/huggingface/transformers.git
!pip install --no-deps --upgrade timm
```

---

## 🏋️ Fine-Tuning Workflow

Run the full fine-tuning pipeline:

```bash
python main.py
```

- Loads and prepares the `Gemma-3N-E4B-it` model
- Loads `FineTome-100k` dataset (3K subset)
- Trains the model using Unsloth’s `SFTTrainer`
- Saves the final model and tokenizer

---

## 🧪 Inference Example

For text-only inference:
```bash
python inference/inference.py
```

For multimodal (image + audio + text) demo:
```bash
python inference/multimodal_demo.py
```

---

## 📦 Export Options

Export your fine-tuned model in multiple formats:

```python
from export.export_model import export_lora, export_merged, export_gguf

# Save LoRA
export_lora(model, tokenizer)

# Save merged FP16
export_merged(model, tokenizer)

# Save GGUF for llama.cpp
export_gguf(model)
```

---

## 📚 Dataset Used

- [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k): A rich collection of multi-turn dialogues used for instruction fine-tuning.

---

## 🧼 Clean Git Setup

Example `.gitignore` content:
```gitignore
# Checkpoints and cache
unsloth_compiled_cache/
*.bin
*.pt
*.safetensors
*.ckpt
*.gguf
*.h5
*.npz
*.log
*.env
*.json
*.yaml

# Environment files
.venv/
__pycache__/
*.egg-info
*.DS_Store
```

---

## 🧠 License & Credits

- Gemma 3N and related weights: [Google & Hugging Face](https://huggingface.co/google/gemma)
- Fine-tuning powered by [Unsloth](https://github.com/unslothai/unsloth)

---

Training complete. You can now push this to GitHub or continue with evaluation and deployment!

> 📘 **Credits:** Installation cells and environment setup snippets were directly adapted from Unsloth's [official documentation](https://github.com/unslothai/unsloth#gemma-3n).

## 👨‍💼 Author

**Elias Hossain**  
_Machine Learning Researcher | PhD Student | AI x Reasoning Enthusiast_

[![GitHub](https://img.shields.io/badge/GitHub-EliasHossain001-blue?logo=github)](https://github.com/EliasHossain001)
