"""
Preprocess pipeline — runs Phase 0 → 1 → 2 on all study images and caches results.

Usage: HF_TOKEN=your_token python preprocess.py [--start N] [--end N]
"""

import argparse
import gc
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from peft import PeftModel
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
    PaliGemmaProcessor,
)

from pipeline.dataset import JSONLDataset
from pipeline.phase0_metadata import run_phase0
from pipeline.phase1_detection import run_phase1
from pipeline.phase2_reasoning import run_phase2
from study_config import (
    COMBINED_IMAGE_COUNT,
    DETECTION_MODEL_ID,
    IMAGE_SIZE,
    REASONING_MODEL_ID,
    TESTSETT_SIZE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_DIR = Path("cache_testsett")

DATASET_SOURCES = [
    {
        "jsonl": Path("/workspace/paligrad/testsett-1/dataset/_annotations.all.jsonl"),
        "images": Path("/workspace/paligrad/testsett-1/dataset"),
        "start_idx": 0,
        "count": TESTSETT_SIZE,
    },
    {
        "jsonl": Path("/workspace/paligrad/Military-4/dataset/_annotations.test.jsonl"),
        "images": Path("/workspace/paligrad/Military-4/dataset"),
        "start_idx": TESTSETT_SIZE,
        "count": COMBINED_IMAGE_COUNT - TESTSETT_SIZE,
    },
]


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_image_list(start: int, end: int) -> list[tuple[int, Image.Image]]:
    """Return [(global_idx, PIL Image), ...] sorted by index."""
    items = []
    for src in DATASET_SOURCES:
        ds = JSONLDataset(src["jsonl"], src["images"])
        for local_i in range(len(ds)):
            global_idx = src["start_idx"] + local_i
            if global_idx < start or global_idx >= end:
                continue
            img, _ = ds[local_i]
            items.append((global_idx, img))
    items.sort(key=lambda x: x[0])
    return items


def p01_cached(img_idx: int) -> bool:
    return (CACHE_DIR / str(img_idx) / "p01_result.json").exists()


def p2_cached(img_idx: int) -> bool:
    return (CACHE_DIR / str(img_idx) / "result.json").exists()


def save_image(img_array: np.ndarray, path: Path):
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=COMBINED_IMAGE_COUNT)
    args = parser.parse_args()

    CACHE_DIR.mkdir(exist_ok=True)

    all_items = build_image_list(args.start, args.end)
    p1_needed = [(idx, img) for idx, img in all_items if not p01_cached(idx)]
    p2_needed = [idx for idx, _ in all_items if not p2_cached(idx)]

    print(f"Range {args.start}–{args.end - 1}  |  Total: {len(all_items)}")
    print(f"Phase 1 needed: {len(p1_needed)}  |  Phase 2 needed: {len(p2_needed)}")

    import os
    hf_token = os.environ.get("HF_TOKEN")

    # Phase 0 + 1
    if p1_needed:
        BASE_MODEL_ID = "google/paligemma2-10b-pt-896"
        print(f"\nLoading PaliFlow ({BASE_MODEL_ID} + LoRA) in 4-bit (nf4)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        processor = PaliGemmaProcessor.from_pretrained(BASE_MODEL_ID, token=hf_token)
        base_model = PaliGemmaForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="eager",
            token=hf_token,
        )
        model_p1 = PeftModel.from_pretrained(base_model, DETECTION_MODEL_ID, token=hf_token)

        # remap adapter keys (peft 0.18.1 naming mismatch)
        adapter_path = hf_hub_download(DETECTION_MODEL_ID, "adapter_model.safetensors", token=hf_token)
        model_state = dict(model_p1.named_parameters())
        loaded = 0
        with safe_open(adapter_path, framework="pt") as f:
            for saved_key in f.keys():
                tensor = f.get_tensor(saved_key)
                remapped = saved_key
                if "base_model.model.model.language_model" in remapped:
                    remapped = remapped.replace(
                        "base_model.model.model.language_model.layers",
                        "base_model.model.language_model.model.layers",
                    )
                elif "base_model.model.model.vision_tower" in remapped:
                    remapped = remapped.replace(
                        "base_model.model.model.vision_tower",
                        "base_model.model.vision_tower",
                    )
                remapped = remapped.replace(".lora_A.weight", ".lora_A.default.weight")
                remapped = remapped.replace(".lora_B.weight", ".lora_B.default.weight")
                if remapped in model_state:
                    model_state[remapped].data.copy_(tensor)
                    loaded += 1
        print(f"Adapter keys remapped: {loaded}/498 loaded")

        model_p1.eval()
        print("PaliFlow loaded.")

        for img_idx, raw_image in p1_needed:
            print(f"\n=== Phase 1 — Image {img_idx} ===")
            img_dir = CACHE_DIR / str(img_idx)
            img_dir.mkdir(exist_ok=True)

            print("  Phase 0 (EXIF + OCR)...")
            p0 = run_phase0(raw_image)

            print("  Phase 1 (detection + heatmaps)...")
            p1 = run_phase1(raw_image, model_p1, processor, DEVICE)

            original_resized = np.array(raw_image.resize((IMAGE_SIZE, IMAGE_SIZE)))
            save_image(original_resized, img_dir / "original.png")
            save_image(p1["bbox_image"], img_dir / "bbox.png")
            for i, hm in enumerate(p1["heatmaps"]):
                save_image(hm["overlay"], img_dir / f"heatmap_{i}.png")

            detected_overlays = [
                hm["overlay"].astype(np.float32)
                for hm in p1["heatmaps"]
                if hm["detected"]
            ]
            if detected_overlays:
                global_hm = np.mean(detected_overlays, axis=0).astype(np.uint8)
                save_image(global_hm, img_dir / "heatmap_global.png")

            p01_data = {
                "exif": p0["exif"],
                "ocr_text": p0["ocr_text"],
                "logos": p0.get("logos"),
                "detections": p1["detections"],
                "heatmaps_meta": [
                    {"label": h["label"], "detected": h["detected"]}
                    for h in p1["heatmaps"]
                ],
            }
            with open(img_dir / "p01_result.json", "w") as f:
                json.dump(p01_data, f, indent=2)

            print(f"  Detections: {[d['label'] for d in p1['detections']] or 'none'}")
            clear_vram()

        print("\nUnloading PaliFlow...")
        del model_p1, base_model, processor
        clear_vram()
    else:
        print("Phase 1: all cached — skipping model load.")

    # Phase 2
    p2_items = [(idx, img) for idx, img in all_items if not p2_cached(idx)]

    if p2_items:
        print(f"\nLoading {REASONING_MODEL_ID} in bfloat16...")
        processor_p2 = AutoProcessor.from_pretrained(REASONING_MODEL_ID, token=hf_token)
        model_p2 = AutoModelForCausalLM.from_pretrained(
            REASONING_MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=hf_token,
        )
        print("Reasoning model loaded.")

        for img_idx, raw_image in p2_items:
            print(f"\n=== Phase 2 — Image {img_idx} ===")
            img_dir = CACHE_DIR / str(img_idx)

            with open(img_dir / "p01_result.json") as f:
                p01 = json.load(f)

            p2 = run_phase2(
                detections=p01["detections"],
                exif=p01["exif"],
                ocr_text=p01["ocr_text"],
                logos=p01.get("logos"),
                model=model_p2,
                processor=processor_p2,
                device=DEVICE,
                image=raw_image,
            )
            print(f"  {p2['classification']} — score {p2['threat_score']}/10")

            result = {
                "image_idx": img_idx,
                "detections": p01["detections"],
                "heatmaps_meta": p01["heatmaps_meta"],
                "exif": p01["exif"],
                "ocr_text": p01["ocr_text"],
                "classification": p2["classification"],
                "threat_score": p2["threat_score"],
                "assessment_reasoning": p2["assessment_reasoning"],
            }
            with open(img_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)

            clear_vram()

        del model_p2, processor_p2
        clear_vram()
    else:
        print("Phase 2: all cached — skipping model load.")

    print(f"\nDone. Cache: {CACHE_DIR}/")


if __name__ == "__main__":
    main()
