"""
Study configuration — 6-group counter-balanced XAI trust study.

Conditions:
  A — Image + threat classification + threat score
  B — A + visual saliency (bbox + attention)
  C — B + textual justification

See GROUP_ROTATION for the full counter-balance design.
"""

# Image sets

# Real participant study: 60 curated images (hand-reviewed via TEST1).
# Categories (FN/FP/BENIGN/???) distributed as evenly as possible across sets.
IMAGE_SETS: dict[int, list[int]] = {
    1: [2, 4, 7, 26, 44, 50, 57, 60, 79, 80, 92, 128, 146, 150, 158, 201, 236, 240, 244, 248],
    2: [0, 6, 23, 55, 64, 70, 71, 81, 83, 123, 149, 163, 165, 166, 190, 211, 217, 224, 227, 237],
    3: [10, 17, 24, 38, 59, 72, 84, 95, 113, 145, 153, 156, 161, 162, 171, 174, 181, 182, 232, 242],
}

# Combined image pool for preprocessing and TEST1:
#   Indices 0–96   → testsett-1 (97 images, already cached)
#   Indices 97–250 → Military-4 test split (154 images)
TESTSETT_SIZE = 97
MILITARY_TEST_SIZE = 154
COMBINED_IMAGE_COUNT = TESTSETT_SIZE + MILITARY_TEST_SIZE  # 251

# TEST1 sees all combined images in a single review pass (condition C = most info)
# Military-4 first (better detection quality), then testsett-1
TEST1_IMAGE_INDICES = list(range(TESTSETT_SIZE, COMBINED_IMAGE_COUNT)) + list(range(TESTSETT_SIZE))

# group → [(round_number, set_number, condition), ...]
# Full permutation design: all 6 orderings of A/B/C.
# Set assignment balanced per-round (each set appears exactly twice per round).
GROUP_ROTATION: dict[int, list[tuple[int, int, str]]] = {
    1: [(1, 1, "A"), (2, 2, "B"), (3, 3, "C")],  # ABC
    2: [(1, 2, "A"), (2, 3, "C"), (3, 1, "B")],  # ACB
    3: [(1, 3, "B"), (2, 1, "A"), (3, 2, "C")],  # BAC
    4: [(1, 1, "B"), (2, 3, "C"), (3, 2, "A")],  # BCA
    5: [(1, 2, "C"), (2, 1, "A"), (3, 3, "B")],  # CAB
    6: [(1, 3, "C"), (2, 2, "B"), (3, 1, "A")],  # CBA
}

GROUP_LABELS: dict[int, str] = {
    1: "ABC", 2: "ACB", 3: "BAC", 4: "BCA", 5: "CAB", 6: "CBA",
}

# 60 real analyst codes — 10 per group
ANALYST_CODES: list[str] = [
    f"G{g}-{n:02d}" for g in range(1, 7) for n in range(1, 11)
]

# QA codes — walk through each group's rotation; results not persisted
TEST_CODES: list[str] = [f"TEST{i}" for i in range(1, 7)]

ALL_CODES: list[str] = ANALYST_CODES + TEST_CODES

CACHE_DIR = "cache_testsett"


def is_test_code(code: str) -> bool:
    return code in TEST_CODES


def get_group(code: str) -> int:
    """Return counter-balance group (1–6) for a participant or test code."""
    if code in TEST_CODES:
        return int(code[-1])  # TEST1 → 1, ..., TEST6 → 6
    return int(code[1])


def get_group_label(code: str) -> str:
    """Return condition-order string (e.g. 'ABC') for the participant's group."""
    return GROUP_LABELS[get_group(code)]


def get_rotation(code: str) -> list[tuple[int, int, str, list[int]]]:
    """
    Return the full rotation for this code as a list of
    (round_number, set_number, condition, image_indices) tuples.

    TEST1 is special: a single pass through all 251 combined images (condition C)
    so the researcher can review every preprocessed image in one session.
    TEST2–TEST6 use their normal group rotations.
    """
    if code == "TEST1":
        return [(1, 0, "C", TEST1_IMAGE_INDICES)]

    group = get_group(code)
    out: list[tuple[int, int, str, list[int]]] = []
    for round_num, set_num, cond in GROUP_ROTATION[group]:
        out.append((round_num, set_num, cond, IMAGE_SETS[set_num]))
    return out


# Pipeline configuration
DETECTION_PROMPT = "detect AirCraft ; Tank\n"
REASONING_MODEL_ID = "google/gemma-3-27b-it"
DETECTION_MODEL_ID = "Snadderr/paligemma-10b-896-military"
IMAGE_SIZE = 896

# Phase 2 prompt guidance (static part; dynamic context assembled in phase2_reasoning.py)
PHASE2_SCORING_GUIDANCE = (
    "Context: These images come from an OSINT collection pipeline monitoring "
    "for military activity.\n"
    "\nScoring guidance:\n"
    "- When no military objects were identified, default strongly to "
    "Non-Threat with a low score (1–3). Only override this if a clearly "
    "identifiable military target is unambiguously visible in the image.\n"
    "- Assess ONLY whether identified objects are military in nature. "
    "Do NOT speculate on the specific model, variant, or designation of "
    "any aircraft or tank — only assess military vs. non-military.\n"
    "- OCR text is optional supplementary context. Do not pivot your "
    "assessment primarily on OCR. If the OCR text appears garbled, "
    "script-misidentified, or incoherent, ignore it entirely.\n"
    "- Logos / insignia are supporting context only, never a dominant "
    "signal. A stray logo alone does not elevate the threat score.\n"
    "- Metadata, when absent, should not affect scoring.\n"
    "- IMPORTANT: Write your response as a unified assessment. Do NOT "
    "reference any 'detector', 'detection system', or 'automated system' — "
    "present your analysis as a single coherent evaluation.\n"
)
