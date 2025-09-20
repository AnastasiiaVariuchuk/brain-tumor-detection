# app.py
import os, io, json, tempfile, time
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

from roboflow import Roboflow
import supervision as sv
import cv2

st.set_page_config(page_title="Brain Tumor Detection", layout="wide")
st.title("🧠 Brain Tumor Detection (Roboflow + Supervision)")

with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Roboflow API Key", value=os.environ.get("ROBOFLOW_API_KEY",""), type="password")
    workspace = st.text_input("Workspace (optional)", value="")
    project_slug = st.text_input("Project slug", value="brain-tumor-8twd6")
    version = st.number_input("Version", min_value=1, value=1, step=1)
    conf = st.slider("Confidence, %", 0, 100, 40, 1)
    overlap = st.slider("Overlap, %", 0, 100, 30, 1)

    st.markdown("---")
    mode = st.radio("Mode", ["Image", "Video"], horizontal=True)

def rf_model():
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace) if workspace else rf.workspace()
    project = ws.project(project_slug)
    return project.version(int(version)).model

def np_rgb(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))

def save_rgb(arr: np.ndarray, path: str):
    Image.fromarray(arr).save(path)

def run_inference_json(model, src_path_or_frame):
    """
    Викликаємо Roboflow і повертаємо JSON.
    - src_path_or_frame: шлях до файлу або RGB-кадр (np.ndarray)
    Примітка: для стабільності у відео ми кодуємо кадр у тимчасовий .jpg.
    """
    if isinstance(src_path_or_frame, str):
        return model.predict(src_path_or_frame, confidence=conf, overlap=overlap).json()

    # RGB np.ndarray -> тимчасовий JPEG -> predict(path)
    rgb = src_path_or_frame
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode frame")
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(buf.tobytes())
        tmp_path = tmp.name
    try:
        return model.predict(tmp_path, confidence=conf, overlap=overlap).json()
    finally:
        try: os.remove(tmp_path)
        except Exception: pass

# ---------- IMAGE MODE ----------
if mode == "Image":
    up = st.file_uploader("📤 Upload image (PNG/JPG)", type=["png","jpg","jpeg"])
    if up and api_key and project_slug:
        try:
            rgb = np_rgb(up)
            with st.status("Loading model & running inference…", expanded=False) as s:
                model = rf_model()
                res = run_inference_json(model, rgb)
                preds = res.get("predictions", [])
                dets = sv.Detections.from_inference(res)
                s.update(label="Done", state="complete")

            st.success(f"Found objects: {len(dets)}")
            classes = sorted({p["class"] for p in preds})
            st.caption(f"Classes: {classes or '—'}")

            sel = st.multiselect("Filter classes", options=classes, default=classes)
            if sel and preds:
                mask = np.array([p["class"] in sel for p in preds])
                dets = dets[mask]
                preds = [p for p, keep in zip(preds, mask) if keep]

            box_annot = sv.BoxAnnotator()
            lbl_annot = sv.LabelAnnotator()
            labels = [f"{p['class']} {p.get('confidence',0)*100:.0f}%" for p in preds]
            annotated = box_annot.annotate(scene=rgb.copy(), detections=dets)
            annotated = lbl_annot.annotate(scene=annotated, detections=dets, labels=labels)

            st.image(annotated, caption="Annotated", use_column_width=True)
            if preds:
                st.dataframe(pd.DataFrame(preds), use_container_width=True)

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as out:
                out_path = out.name
                save_rgb(annotated, out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download annotated PNG", f, file_name="annotated.png")

            with st.expander("Raw JSON"):
                st.code(json.dumps(res, indent=2), "json")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.info("Fill API key, choose project, and upload image.")

# ---------- VIDEO MODE ----------
else:
    upv = st.file_uploader("📤 Upload video (MP4/MOV)", type=["mp4","mov","m4v"])
    col1, col2 = st.columns([1,1])
    with col1:
        fps_limit = st.slider("Process every N-th frame", 1, 10, 1, 1,
                              help=">1 прискорює обробку, але може пропускати дрібні об’єкти")
    with col2:
        draw_traces = st.toggle("Draw traces", value=True)

    # Лінія/лічильник (опційно)
    st.markdown("**Line counter (optional)**")
    lc1, lc2, lc3, lc4 = st.columns(4)
    with lc1: x1 = st.number_input("x1", 0, 9999, 0)
    with lc2: y1 = st.number_input("y1", 0, 9999, 300)
    with lc3: x2 = st.number_input("x2", 0, 9999, 800)
    with lc4: y2 = st.number_input("y2", 0, 9999, 300)

    if upv and api_key and project_slug and st.button("🚀 Start video processing"):
        try:
            # зберігаємо відео у tmp
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmpv:
                tmp_video_in = tmpv.name
                tmpv.write(upv.read())

            # підготовка відео-IO
            info = sv.VideoInfo.from_video_path(tmp_video_in)
            tmp_video_out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            writer = sv.VideoSink(tmp_video_out, video_info=info)

            # трекер / лінія
            byte_tracker = sv.ByteTrack(track_thresh=0.25, track_buffer=30, match_thresh=0.8, frame_rate=max(1, info.fps))
            line_zone = sv.LineZone(start=sv.Point(x1, y1), end=sv.Point(x2, y2))
            line_annot = sv.LineZoneAnnotator(thickness=2, text_thickness=2, text_scale=0.7)

            box_annot = sv.BoxAnnotator()
            trace_annot = sv.TraceAnnotator(thickness=2, trace_length=50)

            model = rf_model()

            st.info("Processing… this may take a while ⏳")
            prog = st.progress(0, text="0%")
            total = int(info.total_frames)
            processed = 0

            for i, frame_bgr in enumerate(sv.get_video_frames_generator(tmp_video_in)):
                if i % fps_limit != 0:
                    continue
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                # інференс + детекції
                res = run_inference_json(model, frame_rgb)
                dets = sv.Detections.from_inference(res)

                # трекінг
                dets = byte_tracker.update_with_detections(dets)

                # підписи (ID + conf)
                labels = []
                conf = dets.confidence if dets.confidence is not None else []
                tids = dets.tracker_id if dets.tracker_id is not None else []
                for j in range(len(dets)):
                    c = conf[j] if j < len(conf) else 0.0
                    tid = tids[j] if j < len(tids) else -1
                    labels.append(f"#{tid} {c*100:.0f}%")

                # анотація
                out = frame_rgb.copy()
                if draw_traces:
                    out = trace_annot.annotate(scene=out, detections=dets)
                out = box_annot.annotate(scene=out, detections=dets, labels=labels)

                # лічильник по лінії (опц.)
                line_zone.trigger(dets)
                out = line_annot.annotate(out, line_counter=line_zone)

                # запис у відеофайл (BGR)
                writer.write_frame(cv2.cvtColor(out, cv2.COLOR_RGB2BGR))

                processed += 1
                if total > 0:
                    prog.progress(min(1.0, processed / total), text=f"{min(100, int(processed/total*100))}%")

            writer.release()
            prog.progress(1.0, text="100%")

            st.success("✅ Video processed.")
            st.video(tmp_video_out)
            with open(tmp_video_out, "rb") as f:
                st.download_button("⬇️ Download annotated MP4", f, file_name="video_out.mp4")

            with st.expander("Line counter stats"):
                st.write(line_zone.in_count, "passed →", line_zone.out_count, "passed ←")

        except Exception as e:
            st.error(f"Video processing failed: {e}")
        finally:
            try:
                if 'tmp_video_in' in locals() and os.path.exists(tmp_video_in): os.remove(tmp_video_in)
            except Exception: pass