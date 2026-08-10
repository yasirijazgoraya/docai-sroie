"""
ARM D inference -- run the FINE-TUNED Qwen2.5-VL (base + LoRA adapter) over the
SAME test split the other arms use, writing to the common schema.

    cd docvlm-rq1
    python -m scripts.04_infer_finetuned --dataset cord --split test
"""
import argparse, json, time
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from src.config import MODEL_BASE, ADAPTER_DIR, OUTPUTS, bnb_config
from src import datasets as ds
from src.parsing import PROMPT, to_record, PROMPT_FUNSD, to_record_funsd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sroie", "cord", "funsd"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--adapter", default=None, help="adapter dir name under outputs/")
    ap.add_argument("--tag", default="qwen_ft", help="output engine tag")
    args = ap.parse_args()

    processor = AutoProcessor.from_pretrained(
        MODEL_BASE, max_pixels=512*28*28, min_pixels=64*28*28)
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_BASE, quantization_config=bnb_config(),
        torch_dtype=torch.bfloat16, device_map="auto")
    adapter_dir = (OUTPUTS / args.adapter) if args.adapter else ADAPTER_DIR
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    samples = ds.load(args.dataset, args.split)
    if args.limit:
        samples = samples[:args.limit]

    out_path = OUTPUTS / f"{args.tag}__{args.dataset}__{args.split}.jsonl"
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, s in enumerate(samples, 1):
            prompt = PROMPT_FUNSD if args.dataset == "funsd" else PROMPT
            messages = [{"role": "user", "content": [
                {"type": "image", "image": str(s.image_path)},
                {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)
            inputs = processor(text=[text], images=image_inputs,
                               padding=True, return_tensors="pt").to(model.device)
            t0 = time.time()
            with torch.no_grad():
                gen = model.generate(**inputs, max_new_tokens=512)
            latency = (time.time() - t0) * 1000
            raw = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:],
                                         skip_special_tokens=True)[0]
            if args.dataset == "funsd":
                rec = to_record_funsd(s.doc_id, raw, latency, engine=args.tag)
            else:
                rec = to_record(s.doc_id, args.dataset, raw, latency, engine=args.tag)
            fh.write(rec.model_dump_json() + "\n")
            if i % 10 == 0:
                print(f"  {i}/{len(samples)}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
