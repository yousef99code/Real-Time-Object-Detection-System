import time
import threading

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer
from ultralytics import YOLO


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="YOLO Vision |Yousef Khaled Madin",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Poppins:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', 'Cairo', sans-serif;
    }
    .stApp {
        background:
          radial-gradient(circle at 12% 8%, rgba(0,224,255,.13), transparent 27%),
          radial-gradient(circle at 88% 18%, rgba(123,97,255,.17), transparent 30%),
          linear-gradient(135deg, #070b1e 0%, #101633 52%, #090d22 100%);
        color: #eef2ff;
    }
    header[data-testid="stHeader"] { background: transparent; }
    section[data-testid="stSidebar"] {
        background: rgba(7, 12, 35, .96);
        border-right: 1px solid rgba(0,224,255,.18);
    }
    .hero {
        text-align: center;
        padding: 22px 8px 18px;
        margin-bottom: 12px;
    }
    .hero-badge {
        display: inline-block;
        padding: 7px 17px;
        border: 1px solid rgba(0,224,255,.45);
        border-radius: 30px;
        color: #62eaff;
        background: rgba(0,224,255,.08);
        letter-spacing: 2px;
        font-size: .78rem;
        animation: pulse 2.5s infinite;
    }
    .hero h1 {
        margin: 12px 0 2px;
        font-size: clamp(2.2rem, 6vw, 4.3rem);
        font-weight: 800;
        background: linear-gradient(90deg, #00e0ff, #8b7bff, #ff55c8, #00e0ff);
        background-size: 250% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shine 7s linear infinite;
    }
    .hero p { color: #c6d2ff; font-size: 1.08rem; margin: 0; }
    @keyframes shine { to { background-position: 250% center; } }
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(0,224,255,.35); }
        70% { box-shadow: 0 0 0 12px rgba(0,224,255,0); }
        100% { box-shadow: 0 0 0 0 rgba(0,224,255,0); }
    }
    .glass {
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 17px;
        padding: 17px 19px;
        min-height: 95px;
        box-shadow: 0 8px 28px rgba(0,0,0,.13);
    }
    .glass h3 { color: #4ce8ff; margin: 0 0 5px; font-size: 1.05rem; }
    .glass p { color: #c8d1f2; margin: 0; font-size: .9rem; }
    .section-title {
        color: #59eaff;
        font-size: 1.05rem;
        font-weight: 700;
        margin: 8px 0;
    }
    .stButton > button, .stDownloadButton > button {
        border: 0;
        border-radius: 11px;
        color: white;
        font-weight: 600;
        background: linear-gradient(90deg, #6657e8, #00bfe8);
        transition: .2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 0 18px rgba(0,224,255,.28);
    }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.055);
        border: 1px solid rgba(255,255,255,.12);
        padding: 12px;
        border-radius: 13px;
    }
    .footer {
        text-align: center;
        color: #9ca9d4;
        margin-top: 30px;
        padding: 13px;
        border-top: 1px solid rgba(255,255,255,.1);
    }
    .footer b { color: #55e7ff; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-badge">⚡ REAL-TIME AI VISION</div>
      <h1>YOLO Vision</h1>
      <p>Live Object Detection &nbsp; • &nbsp; By <b>Yousef Khaled Madin</b></p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Model
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_file: str):
    return YOLO(model_file)


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Detection Settings")
    model_name = st.selectbox(
        "YOLO Model",
        ["yolo26n.pt", "yolo11n.pt", "yolov8n.pt", "yolov8s.pt"],
        index=0,
    )
    confidence = st.slider("Confidence Threshold", 0.05, 1.0, 0.25, 0.05)
    iou = st.slider("IoU Threshold", 0.10, 1.0, 0.70, 0.05)

    st.markdown("---")
    st.markdown("### 🎯 Classes to Detect")
    st.caption("سيب الاختيار فاضي علشان الموديل يكتشف كل الكلاسات.")
    model = load_model(model_name)
    names = model.names
    if isinstance(names, list):
        names = dict(enumerate(names))
    class_options = list(names.values())
    selected_class_names = st.multiselect(
        "Select classes",
        options=class_options,
        default=[],
        placeholder="All classes (default)",
    )
    selected_class_ids = [
        int(class_id)
        for class_id, class_name in names.items()
        if class_name in selected_class_names
    ] if selected_class_names else None

    st.markdown("---")
    st.markdown("### 👨‍💻 Developer")
    st.markdown("**Youssef Madian**")
    st.caption("AI • Computer Vision • Deep Learning")

# ---------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------
def run_detection(image_bgr: np.ndarray):
    result = model.predict(
        source=image_bgr,
        conf=confidence,
        iou=iou,
        classes=selected_class_ids,
        verbose=False,
    )[0]
    annotated_bgr = result.plot()
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

    detections = []
    if result.boxes is not None:
        for class_id, score in zip(
            result.boxes.cls.tolist(),
            result.boxes.conf.tolist(),
        ):
            detections.append({
                "Object": result.names[int(class_id)],
                "Confidence": round(float(score), 3),
            })
    return annotated_rgb, detections


def show_detection_summary(detections):
    if not detections:
        st.warning("No objects detected. Try lowering the confidence threshold.")
        return

    st.success(f"Detected {len(detections)} object(s)")
    counts = {}
    for item in detections:
        counts[item["Object"]] = counts.get(item["Object"], 0) + 1

    columns = st.columns(min(len(counts), 4))
    for index, (label, count) in enumerate(counts.items()):
        columns[index % len(columns)].metric(label, count)

    st.dataframe(detections, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Top feature cards
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        '<div class="glass"><h3>🎥 Live Camera</h3><p>Detect objects directly from your browser camera.</p></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        '<div class="glass"><h3>🖼️ Image Analysis</h3><p>Upload an image and inspect detected classes.</p></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        '<div class="glass"><h3>🎯 Class Filtering</h3><p>Choose exactly which object classes YOLO should detect.</p></div>',
        unsafe_allow_html=True,
    )

st.write("")

# ---------------------------------------------------------
# Browser camera processor
# ---------------------------------------------------------
class YOLOProcessor(VideoProcessorBase):
    def __init__(self):
        self.conf = 0.25
        self.iou = 0.70
        self.classes = None
        self.lock = threading.Lock()

    def update_settings(self, conf, iou_value, classes):
        with self.lock:
            self.conf = conf
            self.iou = iou_value
            self.classes = classes

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")
        with self.lock:
            current_conf = self.conf
            current_iou = self.iou
            current_classes = self.classes

        prediction = model.predict(
            source=image_bgr,
            conf=current_conf,
            iou=current_iou,
            classes=current_classes,
            verbose=False,
        )[0]
        annotated = prediction.plot()
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------
# Main tabs
# ---------------------------------------------------------
tab_live, tab_upload, tab_snapshot = st.tabs(
    ["🎥 Live Browser Camera", "🖼️ Upload Image", "📸 Take Snapshot"]
)

with tab_live:
    st.markdown("### Live Object Detection")
    st.info("اضغط START واسمح للمتصفح باستخدام الكاميرا. المتصفح لازم ياخد إذن منك قبل تشغيلها.")
    ctx = webrtc_streamer(
        key="yolo-vision-live-camera",
        video_processor_factory=YOLOProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    if ctx.video_processor:
        ctx.video_processor.update_settings(
            confidence,
            iou,
            selected_class_ids,
        )
    st.caption(
        "لو الكاميرا مش ظاهرة: تأكد إن المتصفح مسموح له باستخدام الكاميرا، "
        "وإن DroidCam ظاهر ككاميرا في Windows."
    )

with tab_upload:
    st.markdown("### Analyze an Image")
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image)
        if st.button("🔍 Run YOLO Detection", key="detect_upload"):
            with st.spinner("Analyzing image..."):
                start_time = time.time()
                annotated, detections = run_detection(
                    cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                )
                elapsed = (time.time() - start_time) * 1000

            left, right = st.columns(2)
            with left:
                st.markdown("**Original Image**")
                st.image(image_array, use_container_width=True)
            with right:
                st.markdown("**Detection Result**")
                st.image(annotated, use_container_width=True)
            show_detection_summary(detections)
            st.caption(f"Processing time: {elapsed:.0f} ms")

with tab_snapshot:
    st.markdown("### Capture and Analyze")
    snapshot = st.camera_input("Take a picture using your camera")
    if snapshot is not None:
        image = Image.open(snapshot).convert("RGB")
        if st.button("🔍 Analyze Snapshot", key="detect_snapshot"):
            with st.spinner("Analyzing snapshot..."):
                image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                annotated, detections = run_detection(image_bgr)
            st.image(annotated, use_container_width=True)
            show_detection_summary(detections)

st.markdown(
    '<div class="footer">Built with ❤️ using YOLO &amp; Streamlit by <b>Yousef Khaled Madin</b></div>',
    unsafe_allow_html=True,
)
