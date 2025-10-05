# app.py
import os
import json
import hashlib
import tempfile
import platform
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import altair as alt

from roboflow import Roboflow
import supervision as sv


# ----------------------------- Secrets / API Key -----------------------------
def get_roboflow_api_key() -> str:
    """Fetch Roboflow API key from st.secrets, env, or manual input."""
    for k in ("ROBOFLOW_API_KEY", "roboflow_api_key"):
        if k in st.secrets:
            return st.secrets[k]
    if os.getenv("ROBOFLOW_API_KEY"):
        return os.getenv("ROBOFLOW_API_KEY")
    return st.text_input(
        "Enter Roboflow API key",
        type="password",
        help="Not found in st.secrets or env; enter manually for this session.",
    )


# ----------------------------- Streamlit UI ---------------------------------
st.set_page_config(page_title="Brain Tumor App (Detection & Classification)", layout="centered")
st.title("🧠 Brain Tumor App — Detection & Classification")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = get_roboflow_api_key()
    if not api_key:
        st.stop()

    workspace = ""  # empty => default workspace resolved by API key

    task = st.radio("Task", ["Detection", "Classification"], horizontal=True)

    parity_mode = st.toggle(
        "🔬 Parity Mode (debug identical outputs)",
        value=True,
        help="Disables client-side filtering and keeps all detections so local and deployed match."
    )

    if task == "Detection":
        st.subheader("Detection model")
        project_slug_det = st.text_input("Project slug (det)", value="brain-tumor-detection-glu2s")
        version_det = st.number_input("Version (det)", min_value=1, value=1, step=1)
        conf = st.slider("Confidence (%)", 0, 100, 40, 1)
        overlap = st.slider("Overlap (%)", 0, 100, 30, 1)
        display_width = st.slider("Display width (px)", 256, 1280, 768, 16)

        if parity_mode:
            keep_k = 999
            drop_tiny = False
        else:
            keep_k = st.slider("Keep top-K boxes", 1, 50, 5, 1)
            drop_tiny = st.checkbox("Drop tiny boxes (<0.5% image area)", value=True)

    else:
        st.subheader("Classification model")
        project_slug_cls = st.text_input("Project slug (cls)", value="brain-tumor-of8ow")
        version_cls = st.number_input("Version (cls)", min_value=1, value=1, step=1)

uploaded = st.file_uploader("📤 Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])


# ----------------------------- Helpers --------------------------------------
def load_image_to_rgb(file) -> np.ndarray:
    """Read uploaded image → RGB numpy array with EXIF orientation fix."""
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.array(img)


def save_png_exact(arr: np.ndarray, path: str) -> None:
    """Save a deterministic PNG (avoid encoder variations)."""
    Image.fromarray(arr).save(path, format="PNG", optimize=False)


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def rf_model(api_key: str, workspace: str, project_slug: str, version: int):
    """Get Roboflow model by project slug and version."""
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    project = ws.project(project_slug)
    return project.version(int(version)).model


def run_inference_det(model, image_path, confidence: float, overlap: float) -> Dict[str, Any]:
    """Run detection; confidence/overlap are 0..1 fractions."""
    return model.predict(image_path, confidence=confidence, overlap=overlap).json()


def run_inference_cls(model, image_path) -> Dict[str, Any]:
    """Run classification and return JSON."""
    return model.predict(image_path).json()


def normalize_if_needed(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """If values look normalized (<=2), upscale to pixels."""
    if w <= 2.0 and h <= 2.0:
        x *= img_w
        y *= img_h
        w *= img_w
        h *= img_h
    return x, y, w, h


def preds_to_detections(preds: List[Dict[str, Any]], img_w: int, img_h: int) -> Tuple[sv.Detections, Dict[str, int]]:
    """Convert Roboflow preds → supervision.Detections in image pixel space."""
    if not preds:
        return sv.Detections.empty(), {}

    classes = [p.get("class", "object") for p in preds]
    name_to_id = {n: i for i, n in enumerate(sorted(set(classes)))}

    xyxy, conf, cls = [], [], []
    for p in preds:
        x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
        x, y, w, h = normalize_if_needed(x, y, w, h, img_w, img_h)
        x1, y1 = max(0.0, x - w / 2), max(0.0, y - h / 2)
        x2, y2 = min(float(img_w), x + w / 2), min(float(img_h), y + h / 2)
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


# ----------------------------- Main -----------------------------------------
if uploaded and api_key:
    try:
        # Read & persist the exact image that will be sent to Roboflow
        rgb = load_image_to_rgb(uploaded)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            save_png_exact(rgb, tmp_path)
        sha = file_sha256(tmp_path)

        with st.expander("📦 Exact input PNG sent to Roboflow"):
            st.image(tmp_path, caption="Deterministic PNG bytes", use_column_width=True)
            st.code(f"SHA256: {sha}", language="text")

        if task == "Detection":
            st.write("⏳ Loading detection model…")
            model = rf_model(api_key, workspace, project_slug_det, version_det)

            conf_frac = conf / 100.0
            overlap_frac = overlap / 100.0
            st.caption(f"Params → confidence={conf_frac:.2f}, overlap={overlap_frac:.2f}")

            st.write("🚀 Running detection…")
            result = run_inference_det(model, tmp_path, confidence=conf_frac, overlap=overlap_frac)
            raw_preds = result.get("predictions", [])
            meta = result.get("image", {}) or {}

            # Robust image size fallback
            img_h, img_w = rgb.shape[:2]
            try:
                img_w = int(meta.get("width", img_w))
                img_h = int(meta.get("height", img_h))
            except Exception:
                pass

            # Optional tiny-box filter BEFORE further processing
            preds = list(raw_preds)
            if not parity_mode:
                image_area = float(img_w * img_h)
                filtered = []
                for p in preds:
                    x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
                    x, y, w, h = normalize_if_needed(x, y, w, h, img_w, img_h)
                    if drop_tiny and (w * h) / image_area < 0.005:
                        continue
                    filtered.append(dict(p, x=x, y=y, width=w, height=h))
                preds = filtered

                # Sort by confidence and keep top-K
                preds.sort(key=lambda p: p.get("confidence", 0.0), reverse=True)
                preds = preds[:keep_k] if keep_k > 0 else preds

            detections, _ = preds_to_detections(preds, img_w, img_h)
            classes = sorted(set(p.get("class", "object") for p in preds))
            st.success(f"Objects detected: {len(detections)}")
            st.caption(f"Classes: {classes or '—'}")

            # Optional class filter (for usability; parity_mode off preserves raw view)
            if not parity_mode and classes:
                selected = st.multiselect("Filter by classes", options=classes, default=classes)
                if selected and preds:
                    keep_mask = np.array([p.get("class", "object") in selected for p in preds])
                    detections = detections[keep_mask]
                    preds = [p for p, keep in zip(preds, keep_mask) if keep]

            # Render annotated preview
            display_width = int(display_width) if task == "Detection" else 768
            rgb_disp, det_disp, disp_w, disp_h = resize_and_rescale(rgb, detections, display_width)
            labels = [f"{p.get('class','object')} {p.get('confidence',0)*100:.0f}%" for p in preds]
            annotated = sv.BoxAnnotator().annotate(scene=rgb_disp.copy(), detections=det_disp)
            annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=det_disp, labels=labels)
            st.image(annotated, caption=f"Annotated image ({disp_w}×{disp_h})", use_column_width=False, width=disp_w)

            if preds:
                st.dataframe(pd.DataFrame(preds), use_container_width=True)

            # Download exact input and annotated output
            col_a, col_b = st.columns(2)
            with col_a:
                with open(tmp_path, "rb") as f:
                    st.download_button("⬇️ Download exact PNG sent to RF", f, file_name="input_sent_to_roboflow.png")
            with col_b:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
                    out_path = out.name
                    save_png_exact(annotated, out_path)
                with open(out_path, "rb") as f:
                    st.download_button("⬇️ Download annotated PNG", f, file_name="annotated.png")

            with st.expander("🧾 Raw JSON (Roboflow response)"):
                st.code(json.dumps(result, indent=2), language="json")

        else:
            # ------------------- Classification -------------------
            st.write("⏳ Loading classification model…")
            model = rf_model(api_key, workspace, project_slug_cls, version_cls)

            st.write("🧪 Running classification…")
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

                # Top-1
                top_row = df.iloc[df["confidence"].idxmax()] if "confidence" in df else df.iloc[0]
                top_label = str(top_row.get("class", "unknown"))
                top_conf = float(top_row.get("confidence", 0.0)) * 100.0
                st.success(f"Top-1: **{top_label}** ({top_conf:.1f}%)")

                # Show the exact image the classifier saw (same PNG bytes)
                st.image(tmp_path, caption="Input to classifier (deterministic PNG)", use_column_width=True)

                st.dataframe(df, use_container_width=True)

                # Safe Altair chart via Data(values=...) to avoid readonly-layer issue
                if "class" in df and "confidence_%" in df:
                    chart_df = df[["class", "confidence_%"]].copy()
                    chart_df["class"] = chart_df["class"].astype(str)
                    chart_df["confidence_%"] = chart_df["confidence_%"].astype(float)
                    values = chart_df.to_dict(orient="records")
                    auto_height = max(220, 45 * len(values))

                    base = alt.Chart(alt.Data(values=values)).encode(
                        y=alt.Y("class:N", sort="-x", title="Class"),
                        x=alt.X("confidence_%:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
                        tooltip=[
                            alt.Tooltip("class:N", title="Class"),
                            alt.Tooltip("confidence_%:Q", title="Confidence (%)", format=".1f"),
                        ],
                    )

                    bars = base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=0.9)
                    labels = base.mark_text(align="left", dx=6, fontWeight="bold").encode(
                        text=alt.Text("confidence_%:Q", format=".1f")
                    )

                    layered = (bars + labels).properties(height=auto_height)
                    # Configure on the final chart object (LayerChart), not on individual layers
                    layered = layered.configure_axis(grid=True, gridOpacity=0.15, labelFontSize=12, titleFontSize=13)\
                                     .configure_view(strokeOpacity=0)

                    st.altair_chart(layered, use_container_width=True)

                with st.expander("🧾 Raw JSON (Roboflow response)"):
                    st.code(json.dumps(result, indent=2), language="json")

        # ------------------- Environment fingerprint -------------------
        with st.expander("🧪 Environment fingerprint"):
            try:
                import cv2, PIL
                import roboflow as rf_mod
                import supervision as sv_mod
            except Exception:
                cv2 = None; PIL = None; rf_mod = None; sv_mod = None
            st.json({
                "python": platform.python_version(),
                "platform": platform.platform(),
                "roboflow": getattr(rf_mod, "__version__", "unknown"),
                "supervision": getattr(sv_mod, "__version__", "unknown"),
                "pillow": getattr(Image, "__version__", "unknown"),
                "opencv": getattr(cv2, "__version__", "unknown") if cv2 else "unknown",
            })

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