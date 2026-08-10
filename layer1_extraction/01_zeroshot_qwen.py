"""
ARM C -- Qwen2.5-VL ZERO-SHOT extraction.

Runs the base Qwen2.5-VL-7B (no fine-tuning) over the receipt test split, parses
its JSON into the common ExtractionRecord, captures latency, and writes one
JSONL per dataset to outputs/. No scoring here -- that is a separate pass.

    cd docvlm-rq1
    python -m scripts.01_zeroshot_qwen --dataset sroie --split test --limit 50
    python -m scripts.01_zeroshot_qwen --dataset cord  --split test
"""
import argparse, json, time, re
from pathlib import Path

import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

from src.config import MODEL_BASE, OUTPUTS, bnb_config
from src import datasets as ds
from src.parsing import PROMPT, to_record, PROMPT_FUNSD, to_record_funsd




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["sroie", "cord", "funsd"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    print(f"Loading {MODEL_BASE} (4-bit) ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_BASE, quantization_config=bnb_config(),
        torch_dtype=torch.bfloat16, device_map="auto")
    processor = AutoProcessor.from_pretrained(MODEL_BASE)
    model.eval()

    samples = ds.load(args.dataset, args.split)
    if args.limit:
        samples = samples[:args.limit]
    print(f"{len(samples)} samples from {args.dataset}/{args.split}")

    out_path = OUTPUTS / f"zeroshot__{args.dataset}__{args.split}.jsonl"
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
                gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            latency = (time.time() - t0) * 1000
            trimmed = gen[:, inputs.input_ids.shape[1]:]
            raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

            if args.dataset == "funsd":
                rec = to_record_funsd(s.doc_id, raw, latency, engine="qwen_zeroshot")
            else:
                rec = to_record(s.doc_id, args.dataset, raw, latency, engine="qwen_zeroshot")
            fh.write(rec.model_dump_json() + "\n")
            if i % 10 == 0:
                print(f"  {i}/{len(samples)}  ({latency:.0f} ms)")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
