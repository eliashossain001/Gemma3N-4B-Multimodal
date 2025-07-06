# === utils/memory_utils.py ===
import torch

def show_memory_stats():
    gpu_stats = torch.cuda.get_device_properties(0)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    used = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB. Peak reserved = {used} GB.")