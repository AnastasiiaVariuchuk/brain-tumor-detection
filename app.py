import os
import json
import tempfile
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import altair as alt

from roboflow import Roboflow
import supervision as sv

# ---------------- Secrets / API key ----------------
def get_roboflow_api_key() -> str | None:
    # Preferred: Streamlit secrets
    for k in ("ROBOFLOW_API_KEY", "roboflow_api_key"):
        if k in st.secrets:
            return st.secrets[k]
    # Fallback: environment variable
    if os.getenv("ROBOFLOW_API_KEY"):
        return os.getenv("ROBOFLOW_API_KEY")
    # Last resort: prompt user in the sidebar
    return st.text_input(
        "Enter Roboflow API key",
        type="password",
        help="Key not found in st.secrets or environment; enter it for this session.",
    )

# ---------------- UI ----------------
st.set_page_config(page_title="Brain Tumor App (Detection & Classification)", layout="centered")
st.title("🧠 Brain Tumor App — Detection & Classification")

with st.sidebar:
    st.header("⚙️ Settings")

    api_key = get_roboflow_api_key()
    if not api_key:
        st.stop()

    # Leave empty to use default workspace inferred from API key
    workspace = ""

    task = st.radio("Task", ["Detection", "Classification"], horizontal=True)

    if task == "Detection":
        st.subheader("Detection model")
        project_slug_det = st.text_input("Project slug (det)", value="brain-tumor-detection-glu2s")
        version_det = st.number_input("Version (det)", min_value=1, value=1, step=1)
        conf = st.slider("Confidence (%)", 0, 100, 40, 1)
        overlap = st.slider("Overlap (%)", 0, 100, 30, 1)
        display_width = st.slider("Display width (px)", 256, 1024, 768, 16)
        keep_k = st.slider("Keep top-K boxes", 1, 10, 1, 1)
        drop_tiny = st.checkbox("Drop tiny boxes (<0.5% of image area)", value=True)
    else:
        st.subheader("Classification model")
        project_slug_cls = st.text_input("Project slug (cls)", value="brain-tumor-of8ow")
        version_cls = st.number_input("Version (cls)", min_value=1, value=1, step=1)

uploaded = st.file_uploader("📤 Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])

# ---------------- Helpers ----------------
def load_image_to_rgb(file) -> np.ndarray:
    """Read uploaded image -> RGB numpy array with EXIF orientation fix."""
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.array(img)

def rf_model(api_key: str, workspace: str, project_slug: str, version: int):
    """Get Roboflow model by project slug and version."""
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    project = ws.project(project_slug)
    return project.version(int(version)).model

def run_inference_det(model, image_path, confidence: float, overlap: float) -> dict:
    """Run detection prediction and return JSON."""
    return model.predict(image_path, confidence=confidence, overlap=overlap).json()

def run_inference_cls(model, image_path) -> dict:
    """Run classification prediction and return JSON."""
    return model.predict(image_path).json()

def normalize_if_needed(x, y, w, h, img_w, img_h):
    """If coords look normalized (<=2), upscale to pixels."""
    if w <= 2.0 and h <= 2.0:
        x *= img_w; y *= img_h; w *= img_w; h *= img_h
    return x, y, w, h

def preds_to_detections(preds, img_w, img_h):
    """
    Convert predictions (center x,y,width,height in px or normalized)
    to supervision.Detections in the given (img_w, img_h) coordinate space.
    """
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

def scale_image_and_dets(rgb: np.ndarray, detections: sv.Detections, target_w: int):
    """
    Scale a (W,H) image and detections to target_w, preserving aspect ratio.
    If image is already <= target_w, returns original.
    """
    h, w, _ = rgb.shape
    if target_w >= w:
        return rgb, detections, w, h
    scale = target_w / float(w)
    target_h = int(round(h * scale))
    rgb_resized = np.array(Image.fromarray(rgb).resize((target_w, target_h), Image.BILINEAR))
    return rgb_resized, detections.scale((scale, scale)), target_w, target_h

# ---------------- Main ----------------
if uploaded and api_key:
    try:
        # 1) Consistent, lossless input for inference
        rgb_local = load_image_to_rgb(uploaded)

        # Always save PNG (lossless) for Roboflow inference to avoid JPEG artifacts
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
            Image.fromarray(rgb_local).save(tmp_path, format="PNG")

        if task == "Detection":
            st.write("⏳ Loading detection model…")
            model = rf_model(api_key, workspace, project_slug_det, version_det)

            st.write("🚀 Running detection…")
            result = run_inference_det(
                model,
                tmp_path,
                confidence=conf / 100.0,
                overlap=overlap / 100.0,
            )

            # Use model-reported image size to stay in the same coordinate space
            meta = result.get("image", {}) or {}
            model_w = int(meta.get("width", rgb_local.shape[1]))
            model_h = int(meta.get("height", rgb_local.shape[0]))
            raw_preds = result.get("predictions", [])

            # Optional tiny-box filter in model space
            img_area = float(model_w * model_h)
            filtered = []
            for p in raw_preds:
                x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
                x, y, w, h = normalize_if_needed(x, y, w, h, model_w, model_h)
                if drop_tiny and (w * h) / img_area < 0.005:
                    continue
                filtered.append(dict(p, x=x, y=y, width=w, height=h))

            # Keep top-K by confidence for stability
            filtered.sort(key=lambda p: p.get("confidence", 0.0), reverse=True)
            if keep_k > 0:
                filtered = filtered[:keep_k]

            # Build detections in MODEL coordinates
            detections, _ = preds_to_detections(filtered, model_w, model_h)
            classes = sorted(set(p.get("class", "object") for p in filtered))

            st.success(f"Objects detected: {len(detections)}")
            st.caption(f"Classes: {classes or '—'}")

            # Class filter (applies to both preds & detections)
            selected = st.multiselect("Filter by classes", options=classes, default=classes)
            if selected and filtered:
                keep_mask = np.array([p.get("class", "object") in selected for p in filtered])
                detections = detections[keep_mask]
                filtered = [p for p, keep in zip(filtered, keep_mask) if keep]

            # Render: first bring the local image to MODEL size, draw, then scale once for display
            rgb_for_model = np.array(Image.fromarray(rgb_local).resize((model_w, model_h), Image.BILINEAR))
            labels = [f"{p.get('class','object')} {p.get('confidence',0)*100:.0f}%" for p in filtered]

            annot = sv.BoxAnnotator().annotate(scene=rgb_for_model.copy(), detections=detections)
            annot = sv.LabelAnnotator().annotate(scene=annot, detections=detections, labels=labels)

            # One consistent downscale for UI
            annot_disp, _, disp_w, disp_h = scale_image_and_dets(annot, detections, display_width)
            st.image(annot_disp, caption=f"Annotated image ({disp_w}×{disp_h})", width=disp_w, use_column_width=False)

            if filtered:
                st.dataframe(pd.DataFrame(filtered), use_container_width=True)

            # Download annotated image (PNG)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
                out_path = out.name
                Image.fromarray(annot).save(out_path, format="PNG")
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download annotated PNG", f, file_name="annotated.png")

            with st.expander("Raw JSON"):
                st.code(json.dumps(result, indent=2), language="json")

        else:
            # ---------------- Classification ----------------
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
                st.success(f"Top-1: **{str(top_row.get('class', 'unknown'))}** ({float(top_row.get('confidence', 0.0))*100:.1f}%)")

                # Show original image alongside predictions
                st.image(rgb_local, caption="Uploaded image", use_column_width=True)

                # Table
                st.dataframe(df, use_container_width=True)

                # Robust Altair plot (values-based to avoid readonly config issues)
                if {"class", "confidence_%"} <= set(df.columns):
                    plot_df = df[["class", "confidence_%"]].astype({"class": "string", "confidence_%": "float"})
                    values = plot_df.to_dict(orient="records")
                    auto_h = max(220, 44 * len(values))

                    base = alt.Chart(alt.Data(values=values)).encode(
                        y=alt.Y("class:N", sort="-x", title="Class"),
                        x=alt.X("confidence_%:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
                        tooltip=[alt.Tooltip("class:N", title="Class"),
                                 alt.Tooltip("confidence_%:Q", title="Confidence (%)", format=".1f")],
                    )

                    bars = base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=0.9)
                    labels = base.mark_text(align="left", dx=5, fontWeight="bold").encode(
                        text=alt.Text("confidence_%:Q", format=".1f")
                    )

                    chart = (bars + labels).properties(height=auto_h) \
                        .configure_axis(grid=True, gridOpacity=0.15, labelFontSize=12, titleFontSize=13) \
                        .configure_view(strokeOpacity=0)

                    st.altair_chart(chart, use_container_width=True)

                with st.expander("Raw JSON"):
                    st.code(json.dumps(result, indent=2), language="json")

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        try:
            if "tmp_path" in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
else:
    st.info("Upload an image to start.")