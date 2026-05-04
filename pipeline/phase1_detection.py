"""Phase 1 — PaliFlow detection, bounding boxes, and attention heatmaps."""

import re
import numpy as np
import cv2
import torch
from PIL import Image
from scipy.ndimage import median_filter

from study_config import IMAGE_SIZE, DETECTION_PROMPT


# empirical bbox calibration offset after 1024→896 scaling
BBOX_X_OFFSET_PX = 12
BBOX_Y_OFFSET_PX = 0


DETECTION_CLASSES = [
    c.strip() for c in
    DETECTION_PROMPT.replace("detect ", "").replace("\n", "").split(";")
    if c.strip()
]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area


def _extract_heatmap(steps, outputs, img_start, img_end, gen_tokens, image_resized):
    """Extract an attention heatmap from a list of generation steps."""
    all_head_attns = []
    for step in steps:
        if step >= len(outputs.attentions):
            continue
        step_layers = list(outputs.attentions[step])
        if isinstance(step_layers[0], tuple):
            step_layers = [layer[0] for layer in step_layers]

        step_layers = [layer for layer in step_layers if layer is not None]
        if not step_layers:
            continue

        max_seq_len = max(layer.shape[-1] for layer in step_layers)
        for layer in step_layers:
            if layer.shape[-1] == max_seq_len:
                # Keep per-head attention: (num_heads, num_img_tokens)
                img_attn = layer[0, :, 0, img_start:img_end]
                all_head_attns.append(img_attn)

    if not all_head_attns:
        return None

    head_attns = torch.stack(all_head_attns)

    # subtract per-head mean to remove attention sinks
    head_means = head_attns.mean(dim=-1, keepdim=True)
    centered = torch.clamp(head_attns - head_means, min=0)

    final_attn = centered.mean(dim=1).mean(dim=0).to(torch.float32).cpu().numpy()
    heatmap_grid = final_attn.reshape(64, 64)

    heatmap_grid = median_filter(heatmap_grid, size=3)

    v_min = heatmap_grid.min()
    v_max = np.percentile(heatmap_grid, 99.5)
    heatmap_norm = np.clip((heatmap_grid - v_min) / (v_max - v_min + 1e-8), 0, 1)

    heatmap_resized = cv2.resize(heatmap_norm, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(np.array(image_resized), 0.5, heatmap_colored, 0.5, 0)
    return overlay


def run_phase1(image: Image.Image, model, processor, device: str) -> dict:
    """Run Phase 1 detection + heatmap generation on a single image."""
    image_resized = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    prompt = f"<image>\n{DETECTION_PROMPT}"
    inputs = processor(text=prompt, images=image_resized, return_tensors="pt").to(device)

    model.eval()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False,
            output_attentions=True,
            return_dict_in_generate=True,
        )

    # map generated tokens to detected objects, tracking per-class segments
    input_len = inputs["input_ids"].shape[1]
    gen_tokens = outputs.sequences[0][input_len:]

    raw_objects = []
    current_box_tokens = []
    current_label_steps = []

    segments = []
    current_segment = []

    for step, token_id in enumerate(gen_tokens):
        token_str = processor.decode(token_id, skip_special_tokens=False)

        if (
            ";" in token_str
            or "<end>" in token_str
            or "<eos>" in token_str
            or token_id == processor.tokenizer.eos_token_id
        ):
            current_segment.append(step)  # include the separator itself
            segments.append(current_segment)
            current_segment = []

            if len(current_box_tokens) == 4 and len(current_label_steps) > 0:
                raw_objects.append(
                    {"loc_tokens": current_box_tokens, "label_steps": current_label_steps}
                )
            current_box_tokens = []
            current_label_steps = []
            continue

        current_segment.append(step)

        if "<loc" in token_str:
            current_box_tokens.append(token_str)
        else:
            if len(current_box_tokens) == 4:
                current_label_steps.append(step)

    if current_segment:
        segments.append(current_segment)
    if len(current_box_tokens) == 4 and len(current_label_steps) > 0:
        raw_objects.append(
            {"loc_tokens": current_box_tokens, "label_steps": current_label_steps}
        )

    print(f"  Output segments: {len(segments)} (prompt classes: {len(DETECTION_CLASSES)})")

    # NMS filtering
    objects = []
    iou_threshold = 0.2

    for obj in raw_objects:
        coords = [int(re.search(r"\d+", loc).group()) for loc in obj["loc_tokens"]]
        if len(coords) != 4:
            continue

        y_min, x_min, y_max, x_max = coords
        x1 = round((x_min / 1024.0) * IMAGE_SIZE) + BBOX_X_OFFSET_PX
        y1 = round((y_min / 1024.0) * IMAGE_SIZE) + BBOX_Y_OFFSET_PX
        x2 = round((x_max / 1024.0) * IMAGE_SIZE) + BBOX_X_OFFSET_PX
        y2 = round((y_max / 1024.0) * IMAGE_SIZE) + BBOX_Y_OFFSET_PX
        x1 = max(0, min(IMAGE_SIZE - 1, x1))
        y1 = max(0, min(IMAGE_SIZE - 1, y1))
        x2 = max(0, min(IMAGE_SIZE - 1, x2))
        y2 = max(0, min(IMAGE_SIZE - 1, y2))
        box = [x1, y1, x2, y2]
        obj["box"] = box

        keep = True
        for kept_obj in objects:
            if compute_iou(box, kept_obj["box"]) > iou_threshold:
                keep = False
                break
        if keep:
            objects.append(obj)

    print(f"  Found {len(raw_objects)} raw detections, NMS filtered to {len(objects)} unique objects.")

    # locate image tokens
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<image>")
    img_mask = (inputs["input_ids"][0] == image_token_id).cpu().numpy()
    img_indices = np.where(img_mask)[0]

    if len(img_indices) == 4096:
        img_start = img_indices[0]
        img_end = img_indices[-1] + 1
    else:
        img_start = 1 if inputs["input_ids"][0][0].item() == processor.tokenizer.bos_token_id else 0
        img_end = img_start + 4096

    detections = []
    heatmaps = []
    img_with_boxes = np.array(image_resized).copy()

    detected_classes = set()

    for obj in objects:
        box = obj["box"]
        label_text = processor.decode(
            [gen_tokens[i] for i in obj["label_steps"]], skip_special_tokens=True
        ).strip()
        for suffix in ["\n", "<end>", "<eos>", ";", "\n<end>"]:
            label_text = label_text.replace(suffix, "").strip()

        detections.append({"label": label_text, "box": box})
        detected_classes.add(label_text.lower())

        cv2.rectangle(img_with_boxes, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        text_y = max(20, box[1] - 8)
        cv2.putText(
            img_with_boxes, label_text, (box[0], text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )

        overlay = _extract_heatmap(
            obj["label_steps"], outputs, img_start, img_end, gen_tokens, image_resized
        )
        if overlay is not None:
            heatmaps.append({"label": label_text, "overlay": overlay, "detected": True})

    # when nothing detected, generate per-class heatmaps from the relevant segments
    if len(objects) == 0:
        class_segments = {cls.lower(): [] for cls in DETECTION_CLASSES}

        if len(segments) == 1 and len(DETECTION_CLASSES) > 1:
            all_steps = segments[0]
            chunk_size = max(1, len(all_steps) // len(DETECTION_CLASSES))
            for i, cls_name in enumerate(DETECTION_CLASSES):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < len(DETECTION_CLASSES) - 1 else len(all_steps)
                class_segments[cls_name.lower()] = all_steps[start:end]
        else:
            cls_idx = 0
            for seg in segments:
                if cls_idx >= len(DETECTION_CLASSES):
                    break
                current_cls = DETECTION_CLASSES[cls_idx].lower()
                class_segments[current_cls].extend(seg)
                cls_idx += 1

            all_steps = [s for seg in segments for s in seg]
            for cls_name in DETECTION_CLASSES:
                if not class_segments[cls_name.lower()] and all_steps:
                    class_segments[cls_name.lower()] = all_steps

        for cls_name in DETECTION_CLASSES:
            seg_steps = class_segments.get(cls_name.lower(), [])
            if seg_steps:
                overlay = _extract_heatmap(
                    seg_steps, outputs, img_start, img_end, gen_tokens, image_resized
                )
                if overlay is not None:
                    heatmaps.append({
                        "label": f"{cls_name} (not detected)",
                        "overlay": overlay,
                        "detected": False,
                    })
                    print(f"  Generated 'not detected' heatmap for {cls_name} (from {len(seg_steps)} segment tokens)")

    return {
        "detections": detections,
        "bbox_image": img_with_boxes,
        "heatmaps": heatmaps,
    }
