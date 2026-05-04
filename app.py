"""
PaliFlow Study — Streamlit GUI

Usage: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
"""

import base64
import csv
import fcntl
import io
import json
from collections import Counter
from pathlib import Path

import gspread
from google.oauth2.service_account import Credentials
from PIL import Image
import requests
import streamlit as st

st.set_page_config(page_title="PaliFlow Study", layout="centered")
st.markdown(
    "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}</style>",
    unsafe_allow_html=True,
)

from study_config import (
    ALL_CODES,
    CACHE_DIR,
    get_group,
    get_group_label,
    get_rotation,
    is_test_code,
)

CACHE_PATH = Path(CACHE_DIR)
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

DEMO_IDX = 0  # Image used for condition previews in final survey


IMAGE_DATA_COLUMNS = [
    "analyst_id",
    "counter_balance_group",
    "round_number",
    "condition",
    "set_number",
    "image_id",
    "image_order_in_set",
    "vlm_prediction",
    "model_confidence_score",
    "analyst_decision",
    "analyst_confidence",
    "model_insight",
    "explanation_clear",
    "explanation_understood",
]

POST_ROUND_COLUMNS = [
    "analyst_id",
    "round_number",
    "condition",
    "set_number",
    "I_trust_the_model",
    "model_explains_itself",
    "I_understood_model_reasoning",
    "Model_is_helpful",
    "Understandable_explanations",
]

POST_EXPERIMENT_COLUMNS = [
    "analyst_id",
    "group",
    "condition_preference",
    "Workflow_useful",
    "comment_freetext",
    "extra_comments",
]

IMAGE_DATA_PATH = RESULTS_DIR / "image_data.csv"
POST_ROUND_PATH = RESULTS_DIR / "post_round_survey.csv"
POST_EXPERIMENT_PATH = RESULTS_DIR / "post_experiment_survey.csv"

COND_A_SKIP = {"I_trust_the_model", "model_explains_itself", "Understandable_explanations"}

COND_TOOLS = {
    "A": {
        "available": [
            "**Threat classification** (Threat / Non-Threat)",
            "**Threat score** (1–10)",
            "**Image metadata** (location, timestamp, camera)",
        ],
        "unavailable": [
            "Visual saliency (bounding boxes / attention maps)",
            "Textual justification",
        ],
    },
    "B": {
        "available": [
            "**Threat classification** (Threat / Non-Threat)",
            "**Threat score** (1–10)",
            "**Bounding boxes** — rectangles drawn around detected tanks / aircraft",
            "**Attention maps** — heatmaps showing where the model focused",
            "**Image metadata** (location, timestamp, camera)",
        ],
        "unavailable": [
            "Textual justification",
        ],
    },
    "C": {
        "available": [
            "**Threat classification** (Threat / Non-Threat)",
            "**Threat score** (1–10)",
            "**Bounding boxes** — rectangles drawn around detected tanks / aircraft",
            "**Visual saliency maps** — heatmaps showing where the model focused",
            "**Textual justification** — written AI reasoning for the assessment",
            "**Image metadata** (location, timestamp, camera) and detected text/markings",
        ],
        "unavailable": [],
    },
}

SCALE_LABELS = {
    1: "Strongly Disagree",
    2: "Disagree",
    3: "Somewhat Disagree",
    4: "Slightly Disagree",
    5: "Neutral",
    6: "Slightly Agree",
    7: "Somewhat Agree",
    8: "Agree",
    9: "Strongly Agree",
}

EXPLANATION_NOTE = (
    "*Explanations* refers to all context beyond the threat score and "
    "classification itself — i.e. visual elements (bounding boxes, "
    "attention heatmaps) and textual elements (written AI reasoning, "
    "metadata, detected text)."
)


def render_scale(label: str, key: str) -> int | None:
    st.markdown(label)
    cols = st.columns([1, 8, 1])
    with cols[0]:
        st.caption("Strongly Disagree")
    with cols[2]:
        st.caption("Strongly Agree")
    options = ["—"] + list(range(1, 10))
    val = st.select_slider(
        label,
        options=options,
        value="—",
        format_func=lambda x: f"{x} — {SCALE_LABELS[x]}" if isinstance(x, int) else "Select a value",
        key=key,
        label_visibility="collapsed",
    )
    return val if isinstance(val, int) else None


# Helpers

@st.cache_data
def load_cached_result(img_idx: int) -> dict | None:
    result_path = CACHE_PATH / str(img_idx) / "result.json"
    if not result_path.exists():
        return None
    with open(result_path) as f:
        return json.load(f)


def get_image_path(img_idx: int, name: str) -> Path:
    return CACHE_PATH / str(img_idx) / name


@st.cache_data
def load_image_rgb(path: str):
    return Image.open(path).convert("RGB")


@st.cache_data
def load_heatmap_normalized(path: str):
    """Load a heatmap and stretch contrast so the full colour range is used."""
    import numpy as np
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    return Image.fromarray(arr.astype(np.uint8))


def _append_csv(path: Path, columns: list[str], rows: list[dict]):
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=columns)
            if write_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


SHEET_TAB_MAP = {
    "image_data": IMAGE_DATA_COLUMNS,
    "post_round_survey": POST_ROUND_COLUMNS,
    "post_experiment_survey": POST_EXPERIMENT_COLUMNS,
}


_gspread_client = None

def _get_gspread_client():
    global _gspread_client
    if _gspread_client is not None:
        return _gspread_client
    try:
        creds_dict = dict(st.secrets["gcp_service_account"])
        creds = Credentials.from_service_account_info(
            creds_dict,
            scopes=[
                "https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive",
            ],
        )
        _gspread_client = gspread.authorize(creds)
        return _gspread_client
    except Exception:
        return None


def _get_sheet():
    client = _get_gspread_client()
    if client is None:
        return None
    try:
        return client.open_by_key(st.secrets["spreadsheet"]["key"])
    except Exception:
        return None


def _append_sheets(tab_name: str, columns: list[str], rows: list[dict]):
    sheet = _get_sheet()
    if sheet is None:
        raise RuntimeError("Google Sheets connection failed")
    try:
        ws = sheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        ws = sheet.add_worksheet(title=tab_name, rows=1000, cols=len(columns))
        ws.append_row(columns)
    values = [[str(row.get(c, "")) for c in columns] for row in rows]
    ws.append_rows(values)


def already_completed(code: str) -> bool:
    sheet = _get_sheet()
    if sheet is not None:
        try:
            ws = sheet.worksheet("post_experiment_survey")
            col = ws.col_values(1)
            return code in col
        except Exception:
            pass
    if not POST_EXPERIMENT_PATH.exists():
        return False
    with open(POST_EXPERIMENT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        return any(row["analyst_id"] == code for row in reader)


def _rows_to_csv_string(columns: list[str], rows: list[dict]) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def _backup_to_github(code: str):
    try:
        gh = st.secrets["github"]
        token = gh["token"]
        repo = gh["repo"]
        branch = gh.get("branch", "main")
    except (KeyError, TypeError):
        return
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    api = f"https://api.github.com/repos/{repo}"
    files = {
        "results/image_data.csv": (IMAGE_DATA_COLUMNS, st.session_state.image_responses),
        "results/post_round_survey.csv": (POST_ROUND_COLUMNS, st.session_state.round_surveys),
        "results/post_experiment_survey.csv": (POST_EXPERIMENT_COLUMNS, [st.session_state.final_survey]),
    }
    try:
        ref = requests.get(f"{api}/git/ref/heads/{branch}", headers=headers, timeout=10).json()
        base_sha = ref["object"]["sha"]
        base_tree = requests.get(f"{api}/git/commits/{base_sha}", headers=headers, timeout=10).json()["tree"]["sha"]
        tree_items = []
        for path, (columns, rows) in files.items():
            new_rows = _rows_to_csv_string(columns, rows)
            existing = ""
            resp = requests.get(f"{api}/contents/{path}", headers=headers, params={"ref": branch}, timeout=10)
            if resp.status_code == 200:
                existing = base64.b64decode(resp.json()["content"]).decode()
            content = existing + new_rows.split("\n", 1)[1] if existing and new_rows else new_rows
            blob = requests.post(f"{api}/git/blobs", headers=headers, json={
                "content": base64.b64encode(content.encode()).decode(),
                "encoding": "base64",
            }, timeout=10).json()
            tree_items.append({"path": path, "mode": "100644", "type": "blob", "sha": blob["sha"]})
        tree = requests.post(f"{api}/git/trees", headers=headers, json={
            "base_tree": base_tree, "tree": tree_items,
        }, timeout=10).json()
        commit = requests.post(f"{api}/git/commits", headers=headers, json={
            "message": f"Results backup for {code}",
            "tree": tree["sha"],
            "parents": [base_sha],
        }, timeout=10).json()
        requests.patch(f"{api}/git/refs/heads/{branch}", headers=headers, json={
            "sha": commit["sha"],
        }, timeout=10)
    except Exception as e:
        raise RuntimeError(f"GitHub backup failed: {e}") from e


def save_participant_results(code: str):
    errors = []
    try:
        _append_csv(IMAGE_DATA_PATH, IMAGE_DATA_COLUMNS, st.session_state.image_responses)
        _append_csv(POST_ROUND_PATH, POST_ROUND_COLUMNS, st.session_state.round_surveys)
        _append_csv(POST_EXPERIMENT_PATH, POST_EXPERIMENT_COLUMNS, [st.session_state.final_survey])
    except Exception as e:
        errors.append(f"CSV: {e}")
    try:
        _append_sheets("image_data", IMAGE_DATA_COLUMNS, st.session_state.image_responses)
        _append_sheets("post_round_survey", POST_ROUND_COLUMNS, st.session_state.round_surveys)
        _append_sheets("post_experiment_survey", POST_EXPERIMENT_COLUMNS, [st.session_state.final_survey])
    except Exception as e:
        errors.append(f"Sheets: {e}")
    try:
        _backup_to_github(code)
    except Exception as e:
        errors.append(f"GitHub: {e}")
    if errors:
        st.warning(f"Save issues: {'; '.join(errors)}")


# Session state init

if "page" not in st.session_state:
    st.session_state.page = "login"
    st.session_state.code = ""
    st.session_state.rotation = []             # list[(round, set, cond, indices)]
    st.session_state.round_idx = 0             # 0..2
    st.session_state.image_pos = 0             # 0..19 within current round
    st.session_state.image_responses = []      # list[dict] (image_data rows)
    st.session_state.round_surveys = []        # list[dict] (post_round rows)
    st.session_state.final_survey = {}
    st.session_state.final_modality_sel = None  # selected condition in final survey


# Page: Login

def page_login():
    st.title("PaliFlow — Threat Assessment Study")
    st.markdown(
        "Enter your participant access code to begin.\n\n"
        "You will assess **60 images across 3 rounds of 20**. After each "
        "round you will answer a short survey, and a final survey at the end."
    )

    code = st.text_input("Access Code", max_chars=20, key="code_input")

    if st.button("Start Study"):
        code_upper = code.strip().upper()

        if not code_upper:
            st.error("Please enter an access code.")
        elif code_upper not in ALL_CODES:
            st.error("Invalid access code. Please check and try again.")
        elif not is_test_code(code_upper) and already_completed(code_upper):
            st.warning("This code has already been used to submit responses.")
        else:
            st.session_state.code = code_upper
            st.session_state.rotation = get_rotation(code_upper)
            st.session_state.round_idx = 0
            st.session_state.image_pos = 0
            st.session_state.image_responses = []
            st.session_state.round_surveys = []
            st.session_state.final_survey = {}
            st.session_state.final_modality_sel = None
            st.session_state.page = "briefing"
            st.rerun()


# Page: Study Briefing

def page_briefing():
    st.title("Analyst Briefing")
    st.markdown(
        "Please read the following briefing carefully before beginning the study."
    )

    st.markdown("""
### Your Role
You are an intelligence analyst evaluating imagery collected from an open-source
intelligence (OSINT) pipeline. OSINT imagery can come from any source — your
job is not to judge the source, but to assess the **content** of each image and
its context.

Your task is to assess each image for potential military threats, **specifically
focusing on Tanks and Aircraft**. Note that aircraft are inherently significant
in a military monitoring context and should be treated with appropriate concern.

### Study Structure
You will assess **60 images across 3 rounds of 20 images each**. Each round
will present AI output in a different format (see the table below). After each
round you will answer a short survey, and there is a final survey at the end.

### AI Decision Aids
An AI pipeline has pre-processed each image and produced a threat assessment.
Depending on the round, you will have access to different levels of
**explanations** — visual and textual aids that provide further insight into
how the AI arrived at its assessment. You will be told exactly what is
available at the start of each round.

The three possible conditions are:

| Condition | What you will see |
|-----------|-------------------|
| **A** | Threat classification (Threat / Non-Threat) + threat score only |
| **B** | A + **visual explanations**: bounding boxes around detected objects and attention heatmaps |
| **C** | B + **textual justification**: a written explanation of the AI's reasoning |

Below are examples of what each condition looks like in practice.
""")

    demo_result = load_cached_result(DEMO_IDX)
    briefing_conditions = [
        {
            "letter": "A",
            "label": "Condition A — Threat classification + threat score",
            "images": [
                (get_image_path(DEMO_IDX, "original.jpg"), "Original image"),
            ],
            "justification": None,
        },
        {
            "letter": "B",
            "label": "Condition B — A + bounding boxes + attention heatmaps",
            "images": [
                (get_image_path(DEMO_IDX, "bbox.jpg"), "Bounding boxes"),
                (get_image_path(DEMO_IDX, "heatmap_global.jpg"), "Attention heatmap"),
            ],
            "justification": None,
        },
        {
            "letter": "C",
            "label": "Condition C — B + written AI reasoning",
            "images": [
                (get_image_path(DEMO_IDX, "bbox.jpg"), "Bounding boxes"),
                (get_image_path(DEMO_IDX, "heatmap_0.jpg"), "Attention heatmap"),
            ],
            "justification": demo_result,
        },
    ]

    for opt in briefing_conditions:
        with st.container(border=True):
            st.markdown(f"**{opt['label']}**")
            n_extra = 1 if opt["justification"] else 0
            n_cols = len(opt["images"]) + n_extra
            img_cols = st.columns(n_cols)
            for i, (img_path, caption) in enumerate(opt["images"]):
                with img_cols[i]:
                    if img_path.exists():
                        st.image(str(img_path), caption=caption, width=200)
            if opt["justification"]:
                r = opt["justification"]
                clf = r.get("classification", "Unknown")
                score = r.get("threat_score", "?")
                reasoning = r.get("assessment_reasoning", "No reasoning available.")
                first_sentence = reasoning.split(". ")[0] + "." if ". " in reasoning else reasoning
                with img_cols[len(opt["images"])]:
                    st.markdown("**Textual Justification**")
                    st.info(f"**{clf}** — Score: {score}/10\n\n{first_sentence}")

    st.markdown("""
### Important
- You are looking specifically for **military tanks** and **military aircraft**
- The AI threat score runs from **1** (no threat) to **10** (high threat)
- Survey scales run from **1** (Strongly Disagree) to **9** (Strongly Agree)
- The AI output is a **decision aid only** — you may agree or disagree with it
  and should use your own judgement alongside it
- **No further analysis outside this dashboard is necessary.** All the
  information you need to make your assessment is presented on-screen.
- Please set aside any personal feelings towards AI and make your assessments
  as objectively as possible.
- The images may include photographs from **conflict zones**. While none of
  the imagery is graphic, if you find such content upsetting you should not
  proceed with the study.

### ⚠️ Technical Notice
- **Do not refresh the page** — the study will restart from the beginning if you
  do. If anything appears to hang, please wait a moment for it to load.
""")

    if st.button("I have read and understood this briefing — Begin Study", type="primary"):
        st.session_state.page = "round_intro"
        st.rerun()


# Page: Round Introduction

def page_round_intro():
    rotation = st.session_state.rotation
    round_idx = st.session_state.round_idx
    round_num, set_num, condition, _ = rotation[round_idx]

    st.title(f"Round {round_num} of 3")
    st.markdown(
        f"You are about to begin **Round {round_num}** of 3 "
        f"(**Condition {condition}**). "
        "Here is a summary of the AI tools available to you this round."
    )

    tools = COND_TOOLS[condition]
    col_avail, col_unavail = st.columns(2)

    with col_avail:
        st.markdown("#### ✅ Available this round")
        for item in tools["available"]:
            st.markdown(f"- {item}")

    if tools["unavailable"]:
        with col_unavail:
            st.markdown("#### ✗ Not available this round")
            for item in tools["unavailable"]:
                st.markdown(f"- {item}")

    st.info(
        "You are looking for **military tanks** and **military aircraft** specifically. "
        "Assess each image: is there a military threat present?"
    )

    if st.button(f"Begin Round {round_num}", type="primary"):
        st.session_state.page = "assess"
        st.rerun()


# Page: Image Assessment

def render_instructions(condition: str):
    with st.popover("📋 Instructions", use_container_width=True):
        st.markdown("### Analyst Briefing")
        st.markdown(
            "You are an intelligence analyst evaluating imagery collected from "
            "an open-source intelligence (OSINT) pipeline. Your task is to "
            "assess each image for potential military threats, **focusing on "
            "Tanks and Aircraft**. You are looking for threats stemming from "
            "tanks or aircraft."
        )
        tools = COND_TOOLS[condition]
        st.markdown("#### AI Decision Aids Available This Round")
        for item in tools["available"]:
            st.markdown(f"- {item}")
        if tools["unavailable"]:
            st.markdown("#### Not Available This Round")
            for item in tools["unavailable"]:
                st.markdown(f"- ~~{item}~~")
        st.markdown(
            "\nThe AI output is a decision aid — you may agree or disagree "
            "with it. Use it alongside your own judgement."
        )


def render_ai_output(result: dict, condition: str, img_idx: int):
    classification = result.get("classification", "Unknown")
    score = result.get("threat_score", "?")

    def render_classification_box():
        if classification == "Threat":
            st.error(f"**Classification: {classification}**")
        else:
            st.success(f"**Classification: {classification}**")
        st.markdown(f"### THREAT SCORE: {score} / 10")

    def render_metadata():
        exif = result.get("exif") or {}
        meta_lines = []
        if "gps_lat" in exif and "gps_lon" in exif:
            lat = exif["gps_lat"]
            lon = exif["gps_lon"]
            lat_str = f"{abs(lat):.4f}° {'N' if lat >= 0 else 'S'}"
            lon_str = f"{abs(lon):.4f}° {'E' if lon >= 0 else 'W'}"
            meta_lines.append(f"**GPS:** {lat_str}, {lon_str}")
        if exif.get("country"):
            meta_lines.append(f"**Country:** {exif['country']}")
        if exif.get("timestamp"):
            meta_lines.append(f"**Timestamp:** {exif['timestamp']}")
        camera = " ".join(
            part for part in (exif.get("camera_make"), exif.get("camera_model")) if part
        )
        if camera:
            meta_lines.append(f"**Camera:** {camera}")
        if meta_lines:
            st.subheader("Image Metadata")
            st.markdown("  \n".join(meta_lines))

    if condition == "A":
        original_path = get_image_path(img_idx, "original.jpg")
        if original_path.exists():
            st.image(str(original_path), caption="Satellite / OSINT Image", use_container_width=True)
        render_metadata()
        st.subheader("AI Threat Assessment")
        render_classification_box()
        return

    # Condition B or C — render saliency
    col_visual, col_info = st.columns([3, 2]) if condition == "C" else (st.container(), None)

    if condition == "B":
        col_visual = st.container()
        col_info = None

    # Parse per-class heatmap metadata early (needed for toggle logic)
    heatmaps_meta = result.get("heatmaps_meta", result.get("heatmap_labels", []))
    detected_hms = []
    not_detected_hms = []
    for i, hm in enumerate(heatmaps_meta):
        hm_path = get_image_path(img_idx, f"heatmap_{i}.jpg")
        if not hm_path.exists():
            continue
        if isinstance(hm, dict):
            entry = {"path": hm_path, "label": hm["label"], "detected": hm["detected"]}
        else:
            entry = {"path": hm_path, "label": hm, "detected": True}
        if entry["detected"]:
            detected_hms.append(entry)
        else:
            not_detected_hms.append(entry)

    with col_visual:
        bbox_path = get_image_path(img_idx, "bbox.jpg")
        global_path = get_image_path(img_idx, "heatmap_global.jpg")
        has_global = global_path.exists()

        show_heatmap = False
        if has_global or not_detected_hms:
            show_heatmap = st.toggle(
                "Show attention heatmap",
                value=False,
                key=f"heatmap_toggle_{img_idx}",
                help="Switch the main view between detected bounding boxes "
                     "and the model's attention heatmap.",
            )

        if show_heatmap and has_global:
            st.image(
                load_heatmap_normalized(str(global_path)),
                caption="Class-Activation Heatmap (global attention rollout)",
                use_container_width=True,
            )
        elif show_heatmap and not_detected_hms:
            st.image(
                load_heatmap_normalized(str(not_detected_hms[0]["path"])),
                caption="Attention map (no objects detected)",
                use_container_width=True,
            )
        elif bbox_path.exists():
            st.image(str(bbox_path), caption="Detected Objects", use_container_width=True)

        if detected_hms:
            hm_header_cols = st.columns([4, 1])
            with hm_header_cols[0]:
                st.subheader("Detected Object Attention")
            with hm_header_cols[1]:
                with st.popover("ℹ️"):
                    st.markdown(
                        "**About attention maps**\n\n"
                        "These saliency maps show where the model focused "
                        "its attention when identifying each object. Warmer "
                        "colours (red/yellow) indicate higher attention.\n\n"
                        "Attention maps may contain residual noise or "
                        "attention sinks — positions that consistently "
                        "receive high attention regardless of image content."
                    )
            # Number labels when there are multiple detections of the same class
            label_counter: dict[str, int] = {}
            hm_cols = st.columns(min(len(detected_hms), 3))
            for i, hm in enumerate(detected_hms):
                clean_label = hm["label"].replace(" (not detected)", "")
                label_counter[clean_label] = label_counter.get(clean_label, 0) + 1
                # Count total of this class to decide if numbering is needed
                total_of_class = sum(1 for h in detected_hms if h["label"].replace(" (not detected)", "") == clean_label)
                if total_of_class > 1:
                    display_label = f"{clean_label} {label_counter[clean_label]}"
                else:
                    display_label = clean_label
                with hm_cols[i % len(hm_cols)]:
                    st.image(
                        str(hm["path"]),
                        caption=f"Attention when looking for: {display_label}",
                        use_container_width=True,
                    )

        if not_detected_hms:
            st.subheader("Attention Maps (no detection)")
            hm_cols = st.columns(min(len(not_detected_hms), 3))
            for i, hm in enumerate(not_detected_hms):
                clean_label = hm["label"].replace(" (not detected)", "")
                with hm_cols[i % len(hm_cols)]:
                    st.image(
                        str(hm["path"]),
                        caption=f"Attention when looking for: {clean_label}",
                        use_container_width=True,
                    )

    if condition == "B":
        render_metadata()
        st.subheader("AI Threat Assessment")
        render_classification_box()
    elif condition == "C" and col_info is not None:
        with col_info:
            detections = result.get("detections", [])
            if detections:
                st.subheader("Detected Objects")
                counts = Counter(det["label"] for det in detections)
                st.markdown(", ".join(
                    f"{label} × {n}" if n > 1 else label
                    for label, n in counts.items()
                ))
            else:
                st.info("No objects detected.")

            render_metadata()

            ocr_text = result.get("ocr_text")
            logos = result.get("logos")
            ocr_parts = []
            if ocr_text:
                ocr_parts.append(str(ocr_text))
            if logos:
                ocr_parts.append(str(logos))
            if ocr_parts:
                st.subheader("Detected Text & Markings")
                items = [
                    part.strip()
                    for raw in ocr_parts
                    for part in raw.split(";")
                    if part.strip()
                ]
                lines = "<br>".join(items)
                st.markdown(
                    f"<div style='font-size:0.85rem; line-height:1.45'>{lines}</div>",
                    unsafe_allow_html=True,
                )

            st.subheader("AI Threat Assessment")
            render_classification_box()
            reasoning = result.get("assessment_reasoning", "")
            if reasoning:
                st.markdown(reasoning)
    else:
        st.subheader("AI Threat Assessment")
        render_classification_box()


def page_assess():
    rotation = st.session_state.rotation
    round_idx = st.session_state.round_idx
    round_num, set_num, condition, image_indices = rotation[round_idx]
    pos = st.session_state.image_pos
    total = len(image_indices)
    img_idx = image_indices[pos]

    title_col, btn_col = st.columns([4, 1])
    with title_col:
        st.title(f"Round {round_num}/3 — Image {pos + 1} of {total}")
    with btn_col:
        render_instructions(condition)

    # Show cache index for test codes so interesting images can be noted
    if is_test_code(st.session_state.code):
        st.caption(f"🔍 TEST MODE — Cache image index: {img_idx}")

    result = load_cached_result(img_idx)
    if result is None:
        st.error(f"No cached data found for image index {img_idx}. Run preprocess.py first.")
        return

    render_ai_output(result, condition, img_idx)

    # ── Participant response form (fragment: only this reruns on interaction) ──
    @st.fragment
    def assessment_form():
        st.divider()
        st.subheader("Your Assessment")
        st.markdown(
            "No further analysis outside of this dashboard is necessary. "
            "All the information you need is presented above."
        )
        st.caption(EXPLANATION_NOTE)

        decision = st.radio(
            "Based on what you see, how would you classify this image? "
            "Consider your role as a security analyst focusing on the classes **Tanks** and **Aircraft**.",
            options=["Threat", "Non-Threat"],
            index=None,
            key=f"decision_{round_idx}_{pos}",
        )

        analyst_confidence = render_scale(
            "I am confident in my assessment.",
            key=f"conf_{round_idx}_{pos}",
        )

        model_insight = render_scale(
            "I feel that I have insight into how the model arrived at its assessment.",
            key=f"insight_{round_idx}_{pos}",
        )

        explanation_clear = None
        explanation_understood = None
        if condition in ("B", "C"):
            explanation_clear = render_scale(
                "The explanations provided are clear.",
                key=f"clear_{round_idx}_{pos}",
            )

            explanation_understood = render_scale(
                "I understand the explanations provided.",
                key=f"underst_{round_idx}_{pos}",
            )

        if st.button("Submit Assessment", type="primary"):
            if decision is None:
                st.error("Please select a threat classification before submitting.")
                return
            if analyst_confidence is None or model_insight is None:
                st.error("Please answer all scale questions before submitting.")
                return
            if condition in ("B", "C") and (explanation_clear is None or explanation_understood is None):
                st.error("Please answer all explanation questions before submitting.")
                return

            st.session_state.image_responses.append({
                "analyst_id": st.session_state.code,
                "counter_balance_group": get_group(st.session_state.code),
                "round_number": round_num,
                "condition": condition,
                "set_number": set_num,
                "image_id": img_idx,
                "image_order_in_set": pos + 1,
                "vlm_prediction": result.get("classification", ""),
                "model_confidence_score": result.get("threat_score", ""),
                "analyst_decision": decision,
                "analyst_confidence": analyst_confidence,
                "model_insight": model_insight,
                "explanation_clear": explanation_clear if explanation_clear is not None else "",
                "explanation_understood": explanation_understood if explanation_understood is not None else "",
            })

            if pos + 1 < total:
                st.session_state.image_pos += 1
            else:
                st.session_state.page = "round_survey"
            st.rerun()

    assessment_form()


# Page: Per-Round Survey

ROUND_SURVEY_STATEMENTS = {
    "I_trust_the_model": "I generally trust the explanations the model provides",
    "model_explains_itself": "The model explains itself well enough so that I know when to discard its output",
    "I_understood_model_reasoning": "I generally understood why the model made the decision it did, even when wrong",
    "Model_is_helpful": "This model helps me make better decisions",
    "Understandable_explanations": "The explanations are generally easy to understand",
}


def page_round_survey():
    rotation = st.session_state.rotation
    round_idx = st.session_state.round_idx
    round_num, set_num, condition, _ = rotation[round_idx]

    st.title(f"Round {round_num} Survey")
    st.markdown(
        f"You have completed round {round_num} of 3. "
        "Please rate the following statements based on your experience in "
        "this round."
    )
    st.caption(EXPLANATION_NOTE)
    if condition == "A":
        st.info(
            "This round showed AI classification and threat scores only — "
            "no visual or textual explanations. Answer the statements below "
            "based on that experience."
        )

    responses: dict[str, int | str] = {}
    for key, statement in ROUND_SURVEY_STATEMENTS.items():
        if condition == "A" and key in COND_A_SKIP:
            responses[key] = ""
            continue
        responses[key] = render_scale(
            statement,
            key=f"rs_{round_idx}_{key}",
        )

    if st.button("Submit Survey", type="primary"):
        missing = [
            ROUND_SURVEY_STATEMENTS[k]
            for k, v in responses.items()
            if v is None
        ]
        if missing:
            st.error("Please answer all statements before submitting.")
            return

        st.session_state.round_surveys.append({
            "analyst_id": st.session_state.code,
            "round_number": round_num,
            "condition": condition,
            "set_number": set_num,
            **responses,
        })

        if round_idx + 1 < len(rotation):
            st.session_state.round_idx += 1
            st.session_state.image_pos = 0
            st.session_state.page = "round_intro"
        else:
            st.session_state.page = "final_survey"
        st.rerun()


# Page: Final Post-Study Survey

def page_final_survey():
    st.title("Final Survey")
    st.markdown("A few last questions about your overall experience.")

    st.subheader("Which condition did you find most useful overall?")
    st.markdown(
        "Below is a reminder of what each condition looked like. "
        "Click **Select** next to the one you found most useful."
    )

    demo_result = load_cached_result(DEMO_IDX)

    CONDITION_OPTIONS = [
        {
            "letter": "A",
            "label": "Condition A — Threat score only",
            "images": [
                (get_image_path(DEMO_IDX, "original.jpg"), "Original image"),
            ],
            "justification": None,
        },
        {
            "letter": "B",
            "label": "Condition B — Bounding boxes + saliency maps + threat score",
            "images": [
                (get_image_path(DEMO_IDX, "bbox.jpg"), "Bounding boxes"),
                (get_image_path(DEMO_IDX, "heatmap_global.jpg"), "Attention heatmap"),
            ],
            "justification": None,
        },
        {
            "letter": "C",
            "label": "Condition C — Bounding boxes + Visual saliency + textual justification + threat score",
            "images": [
                (get_image_path(DEMO_IDX, "bbox.jpg"), "Bounding boxes"),
                (get_image_path(DEMO_IDX, "heatmap_0.jpg"), "Visual saliency"),
            ],
            "justification": demo_result,
        },
    ]

    for opt in CONDITION_OPTIONS:
        letter = opt["letter"]
        is_sel = st.session_state.final_modality_sel == letter

        with st.container(border=True):
            btn_col, img_col = st.columns([1, 4])

            with btn_col:
                st.markdown(f"**{opt['label']}**")
                btn_label = "✓ Selected" if is_sel else "Select"
                if st.button(
                    btn_label,
                    key=f"sel_{letter}",
                    type="primary" if is_sel else "secondary",
                ):
                    st.session_state.final_modality_sel = letter
                    st.rerun()

            with img_col:
                n_extra = 1 if opt["justification"] else 0
                n_cols = len(opt["images"]) + n_extra
                img_cols = st.columns(n_cols)
                for i, (img_path, caption) in enumerate(opt["images"]):
                    with img_cols[i]:
                        if img_path.exists():
                            st.image(str(img_path), caption=caption, width=160)
                if opt["justification"]:
                    r = opt["justification"]
                    clf = r.get("classification", "Unknown")
                    score = r.get("threat_score", "?")
                    reasoning = r.get("assessment_reasoning", "No reasoning available.")
                    first_sentence = reasoning.split(". ")[0] + "." if ". " in reasoning else reasoning
                    with img_cols[len(opt["images"])]:
                        st.markdown("**Textual Justification**")
                        st.info(f"**{clf}** — Score: {score}/10\n\n{first_sentence}")

    st.divider()

    usefulness = render_scale(
        "To what degree do you think AI-assistance like this or similar can "
        "be useful in workflows like this?",
        key="final_usefulness",
    )

    comments = st.text_area(
        "Why did you prefer the condition you chose above? "
        "Do you think AI assistance like this can be useful in analytical workflows — why or why not?",
        key="final_comments",
    )

    extra_comments = st.text_area(
        "Any other comments? (optional)",
        key="final_extra_comments",
    )

    submitting = st.session_state.get("_submitting", False)
    if st.button("Submit Final Survey", type="primary", disabled=submitting):
        if st.session_state.final_modality_sel is None:
            st.error("Please select the most useful condition.")
            return
        if usefulness is None:
            st.error("Please rate AI-assistance usefulness before submitting.")
            return

        st.session_state._submitting = True

        st.session_state.final_survey = {
            "analyst_id": st.session_state.code,
            "group": get_group_label(st.session_state.code),
            "condition_preference": st.session_state.final_modality_sel,
            "Workflow_useful": usefulness,
            "comment_freetext": comments,
            "extra_comments": extra_comments,
        }

        if not is_test_code(st.session_state.code):
            if not already_completed(st.session_state.code):
                save_participant_results(st.session_state.code)

        st.session_state.page = "done"
        st.rerun()


# Page: Completion

def page_done():
    st.title("Thank You!")
    st.markdown(
        "Your responses have been recorded. "
        "You may now close this page."
    )
    st.balloons()


# Router

pages = {
    "login": page_login,
    "briefing": page_briefing,
    "round_intro": page_round_intro,
    "assess": page_assess,
    "round_survey": page_round_survey,
    "final_survey": page_final_survey,
    "done": page_done,
}

pages[st.session_state.page]()
