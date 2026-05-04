# PaliFlow

A two-pass vision-language pipeline and Streamlit GUI for evaluating analyst trust in AI-generated threat assessments. Built as the artifact for an MSc thesis on Explainable AI (XAI) in OSINT-style military image interpretation.

## Pipeline

Each image flows through three phases, all cached to disk after the first run:

- **Phase 0 — Metadata**: EXIF extraction and OCR (`pipeline/phase0_metadata.py`)
- **Phase 1 — Detection**: Fine-tuned PaliGemma 2 (`Snadderr/paligemma-10b-896-military`) detects military assets (Tank, Soldier, Aircraft) and produces bounding boxes plus per-class attention heatmaps (`pipeline/phase1_detection.py`)
- **Phase 2 — Reasoning**: Gemma 2 27B (4-bit quantized) generates a textual threat justification conditioned on Phase 0 + 1 outputs (`pipeline/phase2_reasoning.py`)

## Study GUI (`app.py`)

A Streamlit application that runs a 6-group counter-balanced within-subject study. Each participant completes three rounds of 20 images, each round under one of three explanation conditions:

- **A** — threat score only
- **B** — A + visual saliency (bounding box + attention heatmap)
- **C** — B + textual justification

Per-image ratings and post-round/post-experiment Likert surveys are written to `results/` and (optionally) backed up to Google Sheets and a private GitHub repo

## Layout

```
app.py                       Streamlit study GUI
preprocess.py                Run phases 0→2 on all study images and cache results
rerun_phase2_metadata.py     Re-run only Phase 2 (e.g. after metadata changes)
pipeline/                    Three-phase inference pipeline
study_config.py              Image sets, group rotation, model IDs
study_metadata.py            Synthesized EXIF for the 60 study images
study_ocr.py                 OCR text for the 60 study images
study_images_60/             60 curated study images, split into 3 sets of 20
cache_testsett/              Pre-computed Phase 0/1/2 outputs (study runs offline)
training/                    PaliGemma 2 fine-tuning notebook
requirements.txt
setup.sh
```

## Running

```bash
bash setup.sh
huggingface-cli login                       # accept Gemma 2 27B license first
python preprocess.py                        # one-time, ~couple hours on A100 80GB
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

Tested on RunPod with an NVIDIA A100 80GB. Phase 1 must run in bfloat16 (do **not** quantize); Phase 2 runs 4-bit quantized.

## Models

- Detection: [`Snadderr/paligemma-10b-896-military`](https://huggingface.co/Snadderr/paligemma-10b-896-military) — fine-tuned PaliGemma 2 10B at 896×896
- Reasoning: [`google/gemma-2-27b-it`](https://huggingface.co/google/gemma-2-27b-it) — license acceptance required
