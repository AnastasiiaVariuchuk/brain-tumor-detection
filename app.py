# app.py — production-clean Streamlit app (no debug visuals)

import os
import json
import hashlib
import tempfile
from typing import Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import altair as alt

from roboflow import Roboflow
import supervision as sv


# ---------------------------------------------------------------------------
# Fixed “prod” settings (can be overridden by st.secrets or environment)
# ---------------------------------------------------------------------------
ROBOFLOW_API_KEY     = os.getenv("ROBOFLOW_API_KEY") or st.secrets.get("ROBOFLOW_API_KEY", "") or "xxxxxx"
ROBOFLOW_WORKSPACE   = os.getenv("ROBOFLOW_WORKSPACE") or st.secrets.get("ROBOFLOW_WORKSPACE", "") or ""
ROBOFLOW_PROJECT_DET = os.getenv("ROBOFLOW_PROJECT_DET") or st.secrets.get("ROBOFLOW_PROJECT_DET", "") or "brain-tumor-detection-glu2s"
ROBOFLOW_VERSION_DET = int(os.getenv("ROBOFLOW_VERSION_DET") or st.secrets.get("ROBOFLOW_VERSION_DET", 1) or 1)

ROBOFLOW_PROJECT_CLS = os.getenv("ROBOFLOW_PROJECT_CLS") or st.secrets.get("ROBOFLOW_PROJECT_CLS", "") or "brain-tumor-of8ow"
ROBOFLOW_VERSION_CLS = int(os.getenv("ROBOFLOW_VERSION_CLS") or st.secrets.get("ROBOFLOW_VERSION_CLS", 1) or 1)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Brain Tumor App (Detection & Classification)", layout="centered")
st.title("🧠 Brain Tumor App — Detection & Classification")

with st.sidebar:
    st.header("⚙️ Settings")
    if not ROBOFLOW_API_KEY or ROBOFLOW_API_KEY == "xxxxxx":
        st.error("Roboflow API key is not set. Add it to environment or `.streamlit/secrets.toml` as `ROBOFLOW_API_KEY`.")
        st.stop()

    task = st.radio("Choose task", ["Detection", "Classification"], horizontal=True)

    if task == "Detection":
        st.subheader("Detection")
        # Model inputs (hidden behind constants, but let user tweak safe params)
        conf = st.slider("Confidence threshold (%)", 0, 100, 40, 1)
        overlap = st.slider("Overlap / NMS (%)", 0, 100, 30, 1)

        st.caption("Display only (doesn't affect predictions)")
        display_width = st.slider("Preview width (px)", 256, 1280, 768, 16)
        keep_k = st.slider("Keep top-K boxes", 1, 50, 5, 1)
        drop_tiny = st.checkbox("Hide tiny boxes (<0.5% image area)", value=True)
        enable_class_filter = st.checkbox("Enable class filter", value=True)

    else:
        st.subheader("Classification")

uploaded = st.file_uploader("📤 Upload an image (PNG/JPG)", type=["png", "jpg", "jpeg"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_image_to_rgb(file) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.array(img)

def save_png_exact(arr: np.ndarray, path: str) -> None:
    Image.fromarray(arr).save(path, format="PNG", optimize=False)

def rf_model(api_key: str, workspace: str, project_slug: str, version: int):
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    project = ws.project(project_slug)
    return project.version(int(version)).model

def run_inference_det(model, image_path, confidence: float, overlap: float) -> Dict[str, Any]:
    return model.predict(image_path, confidence=confidence, overlap=overlap).json()

def run_inference_cls(model, image_path) -> Dict[str, Any]:
    return model.predict(image_path).json()

def normalize_if_needed(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    if w <= 2.0 and h <= 2.0:
        x *= img_w; y *= img_h; w *= img_w; h *= img_h
    return x, y, w, h

def preds_to_detections(preds: List[Dict[str, Any]], img_w: int, img_h: int) -> Tuple[sv.Detections, Dict[str, int]]:
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
    h, w, _ = rgb.shape
    if target_w >= w:
        return rgb, detections, w, h
    scale = target_w / float(w)
    target_h = int(round(h * scale))
    rgb_resized = np.array(Image.fromarray(rgb).resize((target_w, target_h), Image.BILINEAR))
    return rgb_resized, detections.scale((scale, scale)), target_w, target_h


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if uploaded:
    try:
        # Deterministic input PNG (but we don’t show SHA / debug info in prod UI)
        rgb = load_image_to_rgb(uploaded)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            save_png_exact(rgb, tmp_path)

        if task == "Detection":
            model = rf_model(
                api_key=ROBOFLOW_API_KEY,
                workspace=ROBOFLOW_WORKSPACE,
                project_slug=ROBOFLOW_PROJECT_DET,
                version=ROBOFLOW_VERSION_DET,
            )

            result = run_inference_det(
                model, tmp_path,
                confidence=conf / 100.0,
                overlap=overlap / 100.0
            )

            raw_preds = result.get("predictions", [])
            meta = result.get("image", {}) or {}
            img_h, img_w = rgb.shape[:2]
            try:
                img_w = int(meta.get("width", img_w))
                img_h = int(meta.get("height", img_h))
            except Exception:
                pass

            preds = list(raw_preds)

            # display-only: hide tiny, keep top-K
            if drop_tiny and preds:
                image_area = float(img_w * img_h)
                filtered = []
                for p in preds:
                    x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
                    x, y, w, h = normalize_if_needed(x, y, w, h, img_w, img_h)
                    if (w * h) / image_area < 0.005:
                        continue
                    filtered.append(dict(p, x=x, y=y, width=w, height=h))
                preds = filtered

            preds.sort(key=lambda p: p.get("confidence", 0.0), reverse=True)
            preds = preds[:keep_k] if keep_k > 0 else preds

            detections, _ = preds_to_detections(preds, img_w, img_h)
            classes = sorted(set(p.get("class", "object") for p in preds))

            st.success(f"Detections: {len(detections)}")
            if enable_class_filter and classes:
                selected = st.multiselect("Filter by classes", options=classes, default=classes)
                if selected and preds:
                    mask = np.array([p.get("class", "object") in selected for p in preds])
                    detections = detections[mask]
                    preds = [p for p, keep in zip(preds, mask) if keep]

            rgb_disp, det_disp, disp_w, disp_h = resize_and_rescale(rgb, detections, display_width)
            labels = [f"{p.get('class','object')} {p.get('confidence',0)*100:.0f}%" for p in preds]
            annotated = sv.BoxAnnotator().annotate(scene=rgb_disp.copy(), detections=det_disp)
            annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=det_disp, labels=labels)
            st.image(annotated, caption=f"Annotated image ({disp_w}×{disp_h})", use_column_width=False, width=disp_w)

            if preds:
                st.dataframe(pd.DataFrame(preds), use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                with open(tmp_path, "rb") as f:
                    st.download_button("⬇️ Download input PNG", f, file_name="input.png")
            with col_b:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
                    out_path = out.name
                    save_png_exact(annotated, out_path)
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download annotated PNG", f, file_name="annotated.png")

            # Keep raw JSON collapsible, but not expanded and without extra debug text
            with st.expander("Raw JSON"):
                st.code(json.dumps(result, indent=2), language="json")

        else:
            model = rf_model(
                api_key=ROBOFLOW_API_KEY,
                workspace=ROBOFLOW_WORKSPACE,
                project_slug=ROBOFLOW_PROJECT_CLS,
                version=ROBOFLOW_VERSION_CLS,
            )

            result = run_inference_cls(model, tmp_path)
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
                top_conf  = float(top_row.get("confidence", 0.0)) * 100.0
                st.success(f"Top-1: **{top_label}** ({top_conf:.1f}%)")

                st.image(tmp_path, caption="Input image", use_column_width=True)
                st.dataframe(df, use_container_width=True)

                if "class" in df and "confidence_%" in df:
                    values = df[["class", "confidence_%"]].astype({"class": str, "confidence_%": float}).to_dict(orient="records")
                    auto_h = max(220, 45 * len(values))
                    base = alt.Chart(alt.Data(values=values)).encode(
                        y=alt.Y("class:N", sort="-x", title="Class"),
                        x=alt.X("confidence_%:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
                        tooltip=[alt.Tooltip("class:N", title="Class"),
                                 alt.Tooltip("confidence_%:Q", title="Confidence (%)", format=".1f")],
                    )
                    chart = (base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=0.9) +
                             base.mark_text(align="left", dx=6, fontWeight="bold").encode(
                                 text=alt.Text("confidence_%:Q", format=".1f")))
                    chart = chart.properties(height=auto_h).configure_view(strokeOpacity=0).configure_axis(grid=True, gridOpacity=0.15)
                    st.altair_chart(chart, use_container_width=True)

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
else:
    st.info("Upload an image to start.")
