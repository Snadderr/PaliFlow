"""
Re-run Phase 2 for study images with synthesized metadata injected.

Usage: python rerun_phase2_metadata.py
"""

import gc
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from pipeline.phase2_reasoning import run_phase2
from study_config import CACHE_DIR, IMAGE_SETS, REASONING_MODEL_ID
from study_metadata import IMAGE_METADATA
from study_ocr import IMAGE_OCR

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def study_image_indices() -> list[int]:
    indices = set()
    for idxs in IMAGE_SETS.values():
        indices.update(idxs)
    return sorted(indices)


def main():
    hf_token = os.environ.get("HF_TOKEN")
    cache_root = Path(CACHE_DIR)
    indices = study_image_indices()
    print(f"Re-running Phase 2 for {len(indices)} study images")

    print(f"\nLoading reasoning VLM: {REASONING_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(REASONING_MODEL_ID, token=hf_token)
    model = AutoModelForImageTextToText.from_pretrained(
        REASONING_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        token=hf_token,
    )
    print("Reasoning VLM loaded.\n")

    for img_idx in indices:
        img_dir = cache_root / str(img_idx)
        result_path = img_dir / "result.json"
        if not result_path.exists():
            print(f"  [skip] {img_idx}: no result.json")
            continue

        original_path = img_dir / "original.jpg"
        if not original_path.exists():
            original_path = img_dir / "original.png"
        if not original_path.exists():
            print(f"  [skip] {img_idx}: no original image")
            continue

        with open(result_path) as f:
            data = json.load(f)

        metadata = IMAGE_METADATA.get(img_idx)
        ocr_text = IMAGE_OCR.get(img_idx)
        print(f"=== Image {img_idx} === metadata: {metadata}")
        print(f"  ocr: {ocr_text}")

        raw_image = Image.open(original_path).convert("RGB")
        p2 = run_phase2(
            image=raw_image,
            detections=data.get("detections", []),
            exif=metadata,
            ocr_text=ocr_text,
            model=model,
            processor=processor,
            device=DEVICE,
        )
        print(f"  → {p2['classification']} / {p2['threat_score']}/10")

        data["exif"] = metadata
        data["ocr_text"] = ocr_text
        data["classification"] = p2["classification"]
        data["threat_score"] = p2["threat_score"]
        data["assessment_reasoning"] = p2["assessment_reasoning"]

        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)

        clear_vram()

    del model, processor
    clear_vram()
    print("\nPhase 2 metadata re-run complete.")


if __name__ == "__main__":
    main()
