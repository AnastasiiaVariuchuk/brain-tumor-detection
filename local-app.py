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


# ---------------- Helpers ----------------
def get_roboflow_api_key() -> str:
    for k in ("ROBOFLOW_API_KEY", "roboflow_api_key"):
        if k in st.secrets:
            return st.secrets[k]
    if os.getenv("ROBOFLOW_API_KEY"):
        return os.getenv("ROBOFLOW_API_KEY")
    return st.text_input("Enter Roboflow API key", type="password", help="Not found in st.secrets or env. Enter manually for this session.")


# ---------------- UI ----------------
st.set_page_config(page_title="Brain Tumor App (Detection & Classification)", layout="centered")
st.title("🧠 Brain Tumor App — Detection & Classification")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = get_roboflow_api_key()
    if not api_key:
        st.stop()

    workspace = ""
    task = st.radio("Task", ["Detection", "Classification"], horizontal=True)

    if task == "Detection":
        st.subheader("Detection model")
        project_slug_det = st.text_input("Project slug (det)", value="brain-tumor-detection-glu2s")
        version_det = st.number_input("Version (det)", min_value=1, value=1, step=1)
        conf = st.slider("Confidence (%)", 0, 100, 40, 1)
    else:
        st.subheader("Classification model")
        project_slug_cls = st.text_input("Project slug (cls)", value="brain-tumor-of8ow")
        version_cls = st.number_input("Version (cls)", min_value=1, value=1, step=1)

uploaded = st.file_uploader("📤 Upload image (PNG/JPG)", type=["png", "jpg", "jpeg"])


# ---------------- Utility functions ----------------
def load_image_to_rgb(file) -> np.ndarray:
    img = ImageOps.exif_transpose(Image.open(file)).convert("RGB")
    return np.array(img)

def save_rgb_image(arr: np.ndarray, path: str):
    Image.fromarray(arr).save(path)

def rf_model(api_key: str, workspace: str, project_slug: str, version: int):
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    project = ws.project(project_slug)
    return project.version(int(version)).model

def run_inference_det(model, image_path, confidence):
    return model.predict(image_path, confidence=confidence, overlap=0.3).json()

def run_inference_cls(model, image_path):
    return model.predict(image_path).json()

def normalize_if_needed(x, y, w, h, img_w, img_h):
    if w <= 2.0 and h <= 2.0:
        x *= img_w
        y *= img_h
        w *= img_w
        h *= img_h
    return x, y, w, h

def preds_to_detections(preds, img_w, img_h):
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


# ---------------- Main ----------------
if uploaded and api_key:
    try:
        rgb = load_image_to_rgb(uploaded)

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            save_rgb_image(rgb, tmp_path)

        if task == "Detection":
            st.write("⏳ Loading detection model…")
            model = rf_model(api_key, workspace, project_slug_det, version_det)

            st.write("🚀 Running detection…")
            result = run_inference_det(model, tmp_path, confidence=conf / 100.0)
            raw_preds = result.get("predictions", [])

            img_h, img_w = rgb.shape[:2]
            img_area = float(img_w * img_h)
            preds = []
            for p in raw_preds:
                x, y, w, h = float(p["x"]), float(p["y"]), float(p["width"]), float(p["height"])
                x, y, w, h = normalize_if_needed(x, y, w, h, img_w, img_h)
                if (w * h) / img_area < 0.005:
                    continue
                preds.append(dict(p, x=x, y=y, width=w, height=h))

            preds.sort(key=lambda p: p.get("confidence", 0.0), reverse=True)
            preds = preds[:1]  # keep top-1

            detections, _ = preds_to_detections(preds, img_w, img_h)
            classes = sorted(set(p.get("class", "object") for p in preds))

            st.success(f"Objects detected: {len(detections)}")
            st.caption(f"Classes: {classes or '—'}")

            labels = [f"{p.get('class','object')} {p.get('confidence',0)*100:.0f}%" for p in preds]
            annotated = sv.BoxAnnotator().annotate(scene=rgb.copy(), detections=detections)
            annotated = sv.LabelAnnotator().annotate(scene=annotated, detections=detections, labels=labels)

            st.image(annotated, caption=f"Annotated image (768×scaled)", use_column_width=False, width=768)

            if preds:
                st.dataframe(pd.DataFrame(preds), use_container_width=True)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
                out_path = out.name
                save_rgb_image(annotated, out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download annotated PNG", f, file_name="annotated.png")

            with st.expander("Raw JSON"):
                st.code(json.dumps(result, indent=2), language="json")

        else:  # Classification
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

                top_row = df.iloc[df["confidence"].idxmax()] if "confidence" in df else df.iloc[0]
                top_label = str(top_row.get("class", "unknown"))
                top_conf = float(top_row.get("confidence", 0.0)) * 100
                st.success(f"Top-1: **{top_label}** ({top_conf:.1f}%)")

                st.dataframe(df, use_container_width=True)

                if "class" in df and "confidence_%" in df:
                    chart_df = df[["class", "confidence_%"]].copy()
                    chart_df["class"] = chart_df["class"].astype(str)
                    chart_df["confidence_%"] = chart_df["confidence_%"].astype(float)
                    values = chart_df.to_dict(orient="records")
                    auto_height = max(220, 45 * len(values))

                    base = alt.Chart(alt.Data(values=values)).encode(
                        y=alt.Y("class:N", sort="-x", title="Class"),
                        x=alt.X("confidence_%:Q", title="Confidence (%)", scale=alt.Scale(domain=[0, 100])),
                        tooltip=[alt.Tooltip("class:N", title="Class"),
                                 alt.Tooltip("confidence_%:Q", title="Confidence (%)", format=".1f")],
                    )

                    bars = base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, opacity=0.9)
                    labels = base.mark_text(align="left", dx=5, fontWeight="bold")\
                                 .encode(text=alt.Text("confidence_%:Q", format=".1f"))

                    layered = (bars + labels).properties(height=auto_height)\
                                             .configure_axis(grid=True, gridOpacity=0.15, labelFontSize=12, titleFontSize=13)\
                                             .configure_view(strokeOpacity=0)

                    st.altair_chart(layered, use_container_width=True)

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
