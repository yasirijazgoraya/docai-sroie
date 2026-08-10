"""
Central configuration. Edit DATA_ROOT if your paths differ on the server.

On deepblue your layout is:
    /mnt/yasir_drive/E_DATA/data/
        CORD/
        SROIE2019/
        dataset/            <- looks like FUNSD (training_data / testing_data)
        train.pkl dev.pkl test.pkl   <- unknown; inspector will report
"""
from pathlib import Path

DATA_ROOT = Path("/mnt/yasir_drive/E_DATA/data")

PATHS = {
    "sroie": DATA_ROOT / "SROIE2019",
    "cord":  DATA_ROOT / "CORD",
    "funsd": DATA_ROOT / "dataset",
}

# project outputs (relative to this project, NOT the data drive)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# --- model ---
MODEL_BASE = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_DIR = OUTPUTS / "qwen25vl_lora"        # where fine-tuned LoRA weights land

# --- first-pass scope ---
# Receipts (SROIE, CORD) fit the fixed receipt schema below.
# FUNSD is key-value forms (arbitrary keys) -> needs the generic KV schema,
# handled in a second pass. Start with receipts.
RECEIPT_DATASETS = ["sroie", "cord"]

# --- quantization (RTX 4080, 16 GB) ---
# 7B in bf16 (~16 GB weights) will OOM on a 16 GB card, so we load the base in
# 4-bit (QLoRA-style) for inference AND fine-tuning. bitsandbytes required.
USE_4BIT = True


def bnb_config():
    """Shared 4-bit config; returns None if 4-bit disabled."""
    if not USE_4BIT:
        return None
    import torch
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
