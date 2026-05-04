"""Phase 2 — Threat assessment using Gemma 3 27B-it."""

import json
import re

from study_config import PHASE2_SCORING_GUIDANCE


def build_prompt(
    detections: list,
    exif: dict | None,
    ocr_text: str | None,
    logos: str | None = None,
) -> str:
    """Build the Phase 2 prompt from Phase 0 and Phase 1 outputs."""
    if detections:
        label_counts = {}
        for det in detections:
            label = det["label"]
            label_counts[label] = label_counts.get(label, 0) + 1
        objects_str = ", ".join(
            f"{label} (x{count})" if count > 1 else label
            for label, count in label_counts.items()
        )
    else:
        objects_str = "no military objects"

    if exif:
        exif_parts = []
        if "gps_lat" in exif and "gps_lon" in exif:
            lat = exif["gps_lat"]
            lon = exif["gps_lon"]
            lat_str = f"{abs(lat):.4f} {'N' if lat >= 0 else 'S'}"
            lon_str = f"{abs(lon):.4f} {'E' if lon >= 0 else 'W'}"
            exif_parts.append(f"GPS {lat_str}, {lon_str}")
        if "country" in exif:
            exif_parts.append(f"country {exif['country']}")
        if "timestamp" in exif:
            exif_parts.append(f"timestamp {exif['timestamp']}")
        camera_parts = []
        if "camera_make" in exif:
            camera_parts.append(exif["camera_make"])
        if "camera_model" in exif:
            camera_parts.append(exif["camera_model"])
        if camera_parts:
            exif_parts.append(f"camera {' '.join(camera_parts)}")
        metadata_str = "; ".join(exif_parts) if exif_parts else "no metadata available"
    else:
        metadata_str = "no metadata available"

    if ocr_text:
        ocr_str = f'"{ocr_text}"'
    else:
        ocr_str = "no visible text detected"

    prompt = (
        f"You are a military intelligence analyst reviewing imagery from an OSINT collection pipeline.\n"
        f"Objects identified in this image: {objects_str}.\n"
    )

    if exif:
        prompt += f"Image metadata: {metadata_str}.\n"
    if ocr_text:
        prompt += f"Visible text in image: {ocr_str}.\n"
    if logos:
        prompt += f"Visible logos / insignia: \"{logos}\".\n"

    prompt += (
        f"\n{PHASE2_SCORING_GUIDANCE}\n"
        f"Provide:\n"
        f"1. A threat classification: \"Threat\" or \"Non-Threat\"\n"
        f"2. A threat score from 1-10\n"
        f"3. Brief tactical reasoning explaining your assessment\n\n"
        f'Return as JSON: {{"classification": "<Threat or Non-Threat>", "threat_score": <int>, "assessment_reasoning": "<string>"}}'
    )
    return prompt


def parse_response(text: str) -> dict:
    """Extract classification, threat_score and assessment_reasoning from model output."""
    json_match = re.search(r"\{[^{}]*\"threat_score\"[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
            if "threat_score" in result and "assessment_reasoning" in result:
                result["threat_score"] = int(result["threat_score"])
                if "classification" not in result:
                    result["classification"] = "Threat" if result["threat_score"] >= 5 else "Non-Threat"
                return result
        except (json.JSONDecodeError, ValueError):
            pass

    score_match = re.search(r"(?:threat.?score|score)[:\s]*(\d+)", text, re.IGNORECASE)
    score = int(score_match.group(1)) if score_match else 5
    score = min(max(score, 1), 10)

    return {
        "classification": "Threat" if score >= 5 else "Non-Threat",
        "threat_score": score,
        "assessment_reasoning": text.strip(),
    }


def run_phase2(
    detections: list,
    exif: dict | None,
    ocr_text: str | None,
    model,
    processor,
    device: str,
    logos: str | None = None,
    image=None,
) -> dict:
    """Run Phase 2: generate threat assessment from detections + context."""
    user_prompt = build_prompt(detections, exif, ocr_text, logos=logos)

    if image is not None:
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ]},
        ]
    else:
        messages = [
            {"role": "user", "content": user_prompt},
        ]

    inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    import torch
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=False,
        )

    input_len = inputs["input_ids"].shape[1]
    generated = processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
    print(f"  Phase 2 raw output ({len(generated)} chars): {generated[:500]}")

    result = parse_response(generated)
    if not result["assessment_reasoning"]:
        # If parsing produced empty reasoning, use the full output
        print("  WARNING: Empty reasoning from parse, using raw output")
        result["assessment_reasoning"] = generated.strip() if generated.strip() else "No assessment could be generated."
    return result
