# app.py — Production-safe Streamlit app (Roboflow)
# - Uses ONLY st.secrets (no manual keys)
# - Saves temp images as PNG (no JPEG drift)
# - Fixed inference thresholds (0..1) for reproducible results
# - Healthcheck before UI
# - Stable Altair chart rendering

import os
import json
import tempfile
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import altair as alt

from roboflow import Roboflow
import supervision as sv


# =========================
# Strict "deploy" settings
# =========================
DEPLOY = True  # do not change in prod

REQ_SECRETS = ["ROBOFLOW_API_KEY"]
for k in REQ_SECRETS:
    if k not in st.secrets:
        st.error(f"Missing secret: `{k}`. Configure it in the deployment settings.")
        st.stop()

# Optional but recommended secrets; have defaults
WORKSPACE          = st.secrets.get("ROBOFLOW_WORKSPACE", "")
PROJECT_DET        = st.secrets.get("ROBOFLOW_PROJECT_DET", "brain-tumor-detection-glu2s")
VERSION_DET        = int(st.secrets.get("ROBOFLOW_VERSION_DET", "1"))
PROJECT_CLS        = st.secrets.get("ROBOFLOW_PROJECT_CLS", "brain-tumor-of8ow")
VERSION_CLS        = int(st.secrets.get("ROBOFLOW_VERSION_CLS", "1"))

API_KEY = st.secrets["ROBOFLOW_API_KEY"]

# Fixed, reproducible inference params (Roboflow expects 0..1)
CONFIDENCE = 0.40
OVERLAP    = 0.30
KEEP_TOP_K = 10         # keep most confident boxes (no client-side extra filtering)
DROP_TINY  = False      # avoid drifting bboxes by client filters

# =========================
# Small utilities
# =========================
def load_image_to_rgb(file) -> np.ndarray:
    """Read uploaded image -> RGB numpy array with EXIF orientation fix."""
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.array(img)

def save_rgb_image_png(arr: np.ndarray, path: str):
    """Save RGB numpy array as PNG (lossless)."""
    Image.fromarray(arr).save(path, format="PNG")

def rf_model(api_key: str, workspace: str, project_slug: str, version: int):
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    project = ws.project(project_slug)
    return project.version(int(version)).model

def run_inference_det(model, image_path, confidence: float, overlap: float) -> Dict:
    return model.predict(image_path, confidence=confidence, overlap=overlap).json()

def run_inference_cls(model, image_path) -> Dict:
    return model.predict(image_path).json()

def normalize_if_needed(x, y, w, h, img_w, img_h):
    """If values look normalized (<=2), upscale to pixels."""
    if w <= 2.0 and h <= 2.0:
        x *= img_w; y *= img_h; w *= img_w; h *= img_h
    return x, y, w, h

def preds_to_detections(preds, img_w, img_h) -> Tuple[sv.Detections, Dict[str, int]]:
    """Convert center-format predictions to supervision.Detections (xyxy)."""
    if not preds:
        return sv.Detections.empty(), {}
    classes = [p.get("class", "object") for p in preds]
    name_to_id = {n: i for i, n in enumerate(sorted(set(classes)))}

    xyxy, conf, cls = [], [], []
    for p in preds:
        x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
        x, y, w, h = normalize_if_needed(x, y, w, h, img_w, img_h)
        x1, y1 = max(0.0, x - w/2), max(0.0, y - h/2)
        x2, y2 = min(float(img_w), x + w/2), min(float(img_h), y + h/2)
        xyxy.append([x1, y1, x2, y2])
        conf.append(float(p.get("confidence", 0.0)))
        cls.append(name_to_id[p.get("class", "object")])

    return sv.Detections(
        xyxy=np.array(xyxy, dtype=np.float32),
        confidence=np.array(conf, dtype=np.float32),
        class_id=np.array(cls, dtype=np.int32),
    ), name_to_id

def resize_and_rescale(rgb: np.ndarray, detections: sv.Detections, target_w: int):
    """Resize image to target width and scale detections accordingly."""
    h, w, _ = rgb.shape
    if target_w >= w:
        return rgb, detections, w, h
    scale = target_w / float(w)
    target_h = int(round(h * scale))
    rgb_resized = np.array(Image.fromarray(rgb).resize((target_w, target_h), Image.BILINEAR))
    return rgb_resized, detections.scale((scale, scale)), target_w, target_h

def healthcheck():
    """
    Quick check: instantiate model & call a trivial predict on a 1x1 PNG
    to fail early if secrets/workspace/model are wrong.
    """
    try:
        dummy = Image.new("RGB", (1, 1), (0, 0, 0))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            dummy.save(tmp.name, format="PNG")
            m = rf_model(API_KEY, WORKSPACE, PROJECT_DET, VERSION_DET)
            _ = m.predict(tmp.name, confidence=CONFIDENCE, overlap=OVERLAP)
        os.remove(tmp.name)
        return True, ""
    except Exception as e:
        return False, str(e)


# ==============
# Streamlit UI
# ==============
st.set_page_config(page_title="Brain Tumor App (Detection & Classification)", layout="centered")
st.title("🧠 Brain Tumor App — Detection & Classification")

ok, msg = healthcheck()
if not ok:
    st.error(f"Healthcheck failed. Please verify secrets and model settings.\n\nDetails: {msg}")
    st.stop()

with st.sidebar:
    st.header("⚙️ Settings")
    st.caption(f"Workspace: `{WORKSPACE or 'default'}` • DET: `{PROJECT_DET}@v{VERSION_DET}` • CLS: `{PROJECT_CLS}@v{VERSION_CLS}`")
    task = st.radio("Task", ["Detection", "Classification"], horizontal=True)
    display_width = st.slider("Display width (px)", 256, 1024, 768, 16)  # safe visual-only control

uploaded = st.file_uploader("📤 Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if not uploaded:
    st.info("Upload an image to start.")
    st.stop()

# Read and persist as PNG (lossless) for inference
rgb = load_image_to_rgb(uploaded)
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
    tmp_path = tmp.name
    save_rgb_image_png(rgb, tmp_path)

try:
    if task == "Detection":
        st.write("⏳ Loading detection model…")
        det_model = rf_model(API_KEY, WORKSPACE, PROJECT_DET, VERSION_DET)

        st.write("🚀 Running detection…")
        result = run_inference_det(det_model, tmp_path, CONFIDENCE, OVERLAP)
        raw_preds = result.get("predictions", [])

        # No client-side filtering in prod (only KEEP_TOP_K for stable view)
        preds_sorted = sorted(raw_preds, key=lambda p: p.get("confidence", 0.0), reverse=True)
        preds = preds_sorted[:KEEP_TOP_K] if KEEP_TOP_K > 0 else preds_sorted

        img_h, img_w = rgb.shape[:2]
        detections, _ = preds_to_detections(preds, img_w, img_h)
        classes = sorted(set(p.get("class", "object") for p in preds))

        st.success(f"Objects detected: {len(detections)}")
        st.caption(f"Classes: {classes or '—'}")

        rgb_disp, det_disp, disp_w, disp_h = resize_and_rescale(rgb, detections, display_width)
        labels = [f"{p.get('class','object')} {p.get('confidence',0)*100:.0f}%" for p in preds]
        annotated = sv.BoxAnnotator().annotate(scene=rgb_disp.copy(), detections=det_disp)
        annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=det_disp, labels=labels)

        st.image(annotated, caption=f"Annotated image ({disp_w}×{disp_h})", use_column_width=False, width=disp_w)

        if preds:
            # Present the server-returned numbers without post-processing
            st.dataframe(pd.DataFrame(preds), use_container_width=True)

        # Download button
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            out_path = out.name
            save_rgb_image_png(annotated, out_path)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download annotated PNG", f, file_name="annotated.png")

        with st.expander("Raw JSON"):
            st.code(json.dumps(result, indent=2), language="json")

    else:  # Classification
        st.write("⏳ Loading classification model…")
        cls_model = rf_model(API_KEY, WORKSPACE, PROJECT_CLS, VERSION_CLS)

        st.write("🧪 Running classification…")
        result = run_inference_cls(cls_model, tmp_path)
        preds = result.get("predictions", [])

        if not preds:
            st.warning("No classification result.")
        else:
            if isinstance(preds, dict):
                preds = [preds]
            df = pd.DataFrame(preds)

            if "confidence" in df:
                df["confidence_%"] = (df["confidence"] * 100).round(1)

            top_row = df.iloc[df["confidence"].idxmax()] if "confidence" in df else df.iloc[0]
            top_label = str(top_row.get("class", "unknown"))
            top_conf  = float(top_row.get("confidence", 0.0)) * 100
            st.success(f"Top-1: **{top_label}** ({top_conf:.1f}%)")

            # Show original image
            st.image(rgb, caption="Uploaded image", use_column_width=False, width=display_width)

            # Results table
            st.dataframe(df, use_container_width=True)

            # Stable Altair chart (values-based)
            if "class" in df and "confidence_%" in df:
                chart_df = df[["class", "confidence_%"]].copy()
                chart_df["class"] = chart_df["class"].astype(str)
                chart_df["confidence_%"] = chart_df["confidence_%"].astype(float)
                values = chart_df.to_dict(orient="records")
                auto_height = max(220, 45 * len(values))

                base = alt.Chart(alt.Data(values=values)).encode(
                    y=alt.Y("class:N", sort="-x", title="Class"),
                    x=alt.X("confidence_%:Q",
                            title="Confidence (%)",
                            scale=alt.Scale(domain=[0, 100])),
                    tooltip=[
                        alt.Tooltip("class:N", title="Class"),
                        alt.Tooltip("confidence_%:Q", title="Confidence (%)", format=".1f"),
                    ],
                )
                bars = base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=0.9)
                labels = base.mark_text(align="left", dx=5, fontWeight="bold")\
                             .encode(text=alt.Text("confidence_%:Q", format=".1f"))
                st.altair_chart((bars + labels).properties(height=auto_height)
                                .configure_axis(grid=True, gridOpacity=0.15,
                                                labelFontSize=12, titleFontSize=13)
                                .configure_view(strokeOpacity=0),
                                use_container_width=True)

        with st.expander("Raw JSON"):
            st.code(json.dumps(result, indent=2), language="json")

except Exception as e:
    st.error(f"Error: {e}")
finally:
    try:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)
    except Exception:
        pass