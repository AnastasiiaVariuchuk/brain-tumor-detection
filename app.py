import os
import io
import json
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from roboflow import Roboflow
import supervision as sv

# ---------------- UI ----------------
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Detection")
st.caption("Upload an MRI image → get bounding boxes, classes, and model confidence.")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = "162uSH7JuhRNxACBn73k"
    workspace = ""
    project_slug = st.text_input("Project slug", value="brain-tumor-detection-glu2s")
    version = st.number_input("Version", min_value=1, value=1, step=1)

    conf = st.slider("Confidence (%)", 0, 100, 40, 1)
    overlap = st.slider("Overlap (%)", 0, 100, 30, 1)
    display_width = st.slider("Display width (px)", 256, 1024, 768, 16,
                              help="Image is resized for display; boxes are rescaled accordingly.")

uploaded = st.file_uploader("📤 Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])

# ---------------- Helpers ----------------
def load_image_to_rgb(file) -> np.ndarray:
    img = Image.open(file).convert("RGB")
    return np.array(img)

def save_rgb_image(arr: np.ndarray, path: str):
    Image.fromarray(arr).save(path)

def resize_and_rescale_detections(rgb: np.ndarray, detections: sv.Detections, target_w: int):
    """Resize image to target_w (keeping aspect) and scale detections accordingly."""
    h, w, _ = rgb.shape
    if target_w >= w:  # не апскейлимо — залишаємо оригінал
        return rgb, detections, w, h
    scale = target_w / float(w)
    target_h = int(round(h * scale))

    rgb_resized = np.array(Image.fromarray(rgb).resize((target_w, target_h), Image.BILINEAR))
    detections_scaled = detections.scale((scale, scale))  # (x_scale, y_scale)
    return rgb_resized, detections_scaled, target_w, target_h

def run_inference(model, image_path, confidence, overlap):
    """Run inference and show detailed errors if any."""
    try:
        return model.predict(image_path, confidence=confidence, overlap=overlap).json()
    except Exception as e:
        status = None
        body = None
        resp = getattr(e, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
            try:
                body = resp.text
            except Exception:
                body = "<no text>"
        st.error(f"❌ Roboflow request failed. Status: {status}. Error: {e}")
        if body:
            st.code(body, language="json")
        raise

# ---------------- Main ----------------
if uploaded and api_key and project_slug:
    try:
        # 1) Save temporary file (predict uses path)
        rgb = load_image_to_rgb(uploaded)
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            save_rgb_image(rgb, tmp_path)

        # 2) Load model
        st.write("⏳ Loading Roboflow model…")
        rf = Roboflow(api_key=api_key)
        ws = rf.workspace(workspace) if workspace else rf.workspace()
        project = ws.project(project_slug)
        model = project.version(int(version)).model

        # 3) Inference
        st.write("🚀 Running inference…")
        result = run_inference(model, tmp_path, conf, overlap)
        preds = result.get("predictions", [])

        # 4) Detections
        detections = sv.Detections.from_inference(result)
        classes_in_result = sorted({p["class"] for p in preds})
        st.success(f"Objects detected: {len(detections)}")
        st.write("Classes:", classes_in_result if classes_in_result else "—")

        # 5) Optional class filter
        selected = st.multiselect("Filter by classes", options=classes_in_result, default=classes_in_result)
        if selected and preds:
            mask = np.array([p["class"] in selected for p in preds])
            detections = detections[mask]
            preds = [p for p, keep in zip(preds, mask) if keep]

        # 6) Resize image for display + rescale boxes to match
        rgb_disp, detections_disp, disp_w, disp_h = resize_and_rescale_detections(rgb, detections, display_width)

        # 7) Annotate
        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()
        labels = [f"{p['class']} {p.get('confidence', 0)*100:.0f}%" for p in preds]

        annotated = box_annotator.annotate(scene=rgb_disp.copy(), detections=detections_disp)
        annotated = label_annotator.annotate(scene=annotated, detections=detections_disp, labels=labels)

        # 8) Show result (no auto-resize by Streamlit)
        st.image(annotated, caption=f"Annotated image ({disp_w}×{disp_h})", use_column_width=False, width=disp_w)

        # Detection table
        if preds:
            df = pd.DataFrame(preds)
            st.dataframe(df, use_container_width=True)

        # 9) Download annotated PNG (use displayed image)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
            out_path = out.name
            save_rgb_image(annotated, out_path)
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download annotated PNG", f, file_name="annotated.png")

        # 10) Raw JSON
        with st.expander("📦 Raw JSON response"):
            st.code(json.dumps(result, indent=2), language="json")

    except Exception as e:
        st.error(f"Execution error: {e}")
    finally:
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
else:
    st.info("Please upload an image.")
