"""
streamlit.py
----------------
Streamlit web application for Real-Time Object Detection & Tracking.
 
Features:
  • Upload a video file → process frame-by-frame with live preview
  • Live webcam stream via streamlit-webrtc (WebRTC)
  • Full sidebar controls: model, confidence, classes, tracker
  • Per-class object count panel
  • Download processed video
  • Stop button mid-processing
 
Run:
    streamlit run streamlit.py
"""
 
from __future__ import annotations
 
import os
import cv2
import time
import tempfile
import numpy as np
import streamlit as st
from pathlib import Path
 
from detector import ObjectDetector
from tracker import TrackerFactory, SORTTracker
from utils import (
    draw_track,
    draw_fps,
    draw_object_count,
    format_label,
    FPSCounter,
    ObjectCounter,
)
 
# ── Optional WebRTC (for live webcam tab) ───────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
 
 
# ═══════════════════════════════════════════════════════════════════════
# Page configuration
# ═══════════════════════════════════════════════════════════════════════
 
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        # Setting these to None removes the corresponding entries from the
        # hamburger menu (Get help / Report a bug / About)
        "Get Help": None,
        "Report a bug": None,
        "About": None,
    },
)
 
# ── Custom CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');
 
  /* ══════════════════════════════════════════════════════════════════
     Hide Streamlit Cloud "GitHub" viewer badge & "Fork this app" menu
     (keeps your repo / personal GitHub profile out of view)
     ══════════════════════════════════════════════════════════════════ */
 
  /* GitHub avatar / "View source" badge in the top-right corner */
  [class*="_viewerBadge_"],
  [class*="viewerBadge_"],
  [class*="_profileContainer_"],
  [class*="_profilePreview_"],
  [class*="_link_"][href*="github.com"],
  a[href*="streamlit.io/cloud"],
  a[href*="share.streamlit.io"][class*="viewerBadge"] {
    display: none !important;
    visibility: hidden !important;
  }
 
  /* Hamburger "⋮" main menu — contains "Fork this app" / "View source" */
  #MainMenu,
  [data-testid="stMainMenu"],
  [data-testid="stToolbarActions"],
  [data-testid="stDeployButton"] {
    display: none !important;
    visibility: hidden !important;
  }
 
  /* "Made with Streamlit" footer */
  footer { visibility: hidden !important; height: 0 !important; }
  footer:after { display: none !important; }
 
  /* Decorative emojis to fill the empty top-right space */
  [data-testid="stHeader"] {
    position: relative;
    background: transparent !important;
  }
  [data-testid="stHeader"]::after {
    content: "🎯  ✨  🚀  💜  ⚡";
    position: fixed;
    top: 14px;
    right: 22px;
    font-size: 18px;
    letter-spacing: 4px;
    z-index: 999999;
    pointer-events: none;
    filter: drop-shadow(0 0 8px rgba(124,58,237,0.55));
    animation: emojifloat 3.5s ease-in-out infinite;
  }
  @keyframes emojifloat {
    0%, 100% { transform: translateY(0); opacity: 0.95; }
    50%      { transform: translateY(-2px); opacity: 1; }
  }
 
  /* ── Base: deep space background with mesh ── */
  .stApp {
    background-color: #060b18;
    background-image:
      radial-gradient(ellipse at 20% 20%, rgba(124,58,237,0.12) 0%, transparent 50%),
      radial-gradient(ellipse at 80% 10%, rgba(37,99,235,0.10) 0%, transparent 45%),
      radial-gradient(ellipse at 60% 80%, rgba(8,145,178,0.08) 0%, transparent 50%);
    background-attachment: fixed;
  }
 
  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0c1022 0%, #0a0e1c 100%);
    border-right: 1px solid rgba(124,58,237,0.2);
  }
  section[data-testid="stSidebar"] .stMarkdown p,
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stSlider label,
  section[data-testid="stSidebar"] .stMultiSelect label,
  section[data-testid="stSidebar"] .stTextInput label,
  section[data-testid="stSidebar"] .stToggle label {
    color: #ffffff !important;
  }
 
  /* ── Metric cards ── */
  div[data-testid="metric-container"] {
    background: linear-gradient(135deg, rgba(124,58,237,0.08) 0%, rgba(37,99,235,0.06) 100%);
    border: 1px solid rgba(124,58,237,0.25);
    border-top: 2px solid #7c3aed;
    border-radius: 14px;
    padding: 14px 18px;
  }
  div[data-testid="metric-container"] label {
    color: #ff6b35 !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1.2px;
  }
  div[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #a78bfa !important;
    font-size: 1.7rem !important;
    font-weight: 800 !important;
  }
 
  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: rgba(12,16,34,0.9);
    border-radius: 14px;
    padding: 5px;
    border: 1px solid rgba(124,58,237,0.15);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #ffb347 !important;
    font-weight: 600;
    font-size: 0.88rem;
    padding: 8px 22px;
    transition: all 0.25s;
  }
  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%) !important;
    color: white !important;
    box-shadow: 0 0 20px rgba(124,58,237,0.45);
  }
 
  /* ── Buttons ── */
  .stButton > button {
    background: linear-gradient(135deg, #7c3aed 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 10px 28px !important;
    letter-spacing: 0.04em;
    box-shadow: 0 4px 20px rgba(124,58,237,0.35);
    transition: all 0.25s !important;
  }
  .stButton > button:hover {
    box-shadow: 0 6px 30px rgba(124,58,237,0.6) !important;
    transform: translateY(-2px) !important;
  }
 
  /* ── Download button ── */
  .stDownloadButton > button {
    background: linear-gradient(135deg, #0891b2 0%, #0d9488 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    box-shadow: 0 4px 18px rgba(8,145,178,0.4) !important;
    transition: all 0.25s !important;
  }
  .stDownloadButton > button:hover {
    box-shadow: 0 6px 26px rgba(8,145,178,0.65) !important;
    transform: translateY(-2px) !important;
  }
 
  /* ── Progress bar ── */
  .stProgress > div > div {
    background: linear-gradient(90deg, #7c3aed, #2563eb, #06b6d4) !important;
    border-radius: 99px;
    box-shadow: 0 0 10px rgba(124,58,237,0.4);
  }
  .stProgress > div {
    background: rgba(124,58,237,0.1) !important;
    border-radius: 99px;
  }
 
  /* ── File uploader ── */
  [data-testid="stFileUploader"] {
    border: 2px dashed rgba(124,58,237,0.3) !important;
    border-radius: 16px !important;
    background: rgba(124,58,237,0.03) !important;
    transition: border-color 0.3s, background 0.3s;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: rgba(124,58,237,0.55) !important;
    background: rgba(124,58,237,0.06) !important;
  }
 
  /* ── Alerts ── */
  .stAlert { border-radius: 12px !important; }
 
  /* ── Dividers ── */
  hr { border-color: rgba(124,58,237,0.12) !important; }
 
  /* ── Selectbox / inputs — dark background ── */
  [data-baseweb="select"] { border-radius: 10px !important; }
  [data-baseweb="select"] > div:first-child {
    background: #111827 !important;
    border: 1px solid rgba(255,107,53,0.3) !important;
    color: #ffffff !important;
  }
  [data-baseweb="select"] [data-testid="stSelectbox"] div,
  [data-baseweb="select"] span {
    color: #ffffff !important;
  }
  /* Slider labels and values */
  .stSlider [data-testid="stTickBarMin"],
  .stSlider [data-testid="stTickBarMax"],
  .stSlider p {
    color: #ff6b35 !important;
  }
  /* Text inputs */
  .stTextInput input {
    background: #111827 !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,107,53,0.3) !important;
  }
  /* Multiselect */
  [data-baseweb="tag"] {
    background: rgba(255,107,53,0.2) !important;
  }
  [data-baseweb="tag"] span { color: #ff6b35 !important; }
 
  /* ── Stat box (custom cards) ── */
  .stat-box {
    background: linear-gradient(135deg, rgba(124,58,237,0.08), rgba(37,99,235,0.06));
    border: 1px solid rgba(124,58,237,0.2);
    border-radius: 14px;
    padding: 18px 16px;
    margin-bottom: 12px;
    text-align: center;
    transition: transform 0.2s, box-shadow 0.2s;
  }
  .stat-box:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 32px rgba(124,58,237,0.18);
  }
  .stat-title {
    color: #ff6b35;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    font-weight: 600;
  }
  .stat-value {
    color: #a78bfa;
    font-size: 34px;
    font-weight: 800;
    margin: 6px 0 2px 0;
    text-shadow: 0 0 24px rgba(167,139,250,0.35);
  }
  .stat-sub { color: #34d399; font-size: 12px; font-weight: 600; }
 
  /* ── Class badges (multi-color cycling) ── */
  .class-badge {
    display: inline-block;
    border-radius: 99px;
    padding: 5px 14px;
    margin: 4px;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
  }
  .cb-purple {
    background: linear-gradient(135deg, #7c3aed, #6d28d9);
    color: white;
    box-shadow: 0 2px 10px rgba(124,58,237,0.4);
  }
  .cb-blue {
    background: linear-gradient(135deg, #2563eb, #1d4ed8);
    color: white;
    box-shadow: 0 2px 10px rgba(37,99,235,0.4);
  }
  .cb-teal {
    background: linear-gradient(135deg, #0891b2, #0d9488);
    color: white;
    box-shadow: 0 2px 10px rgba(8,145,178,0.4);
  }
  .cb-amber {
    background: linear-gradient(135deg, #d97706, #b45309);
    color: white;
    box-shadow: 0 2px 10px rgba(217,119,6,0.4);
  }
  .cb-rose {
    background: linear-gradient(135deg, #e11d48, #be123c);
    color: white;
    box-shadow: 0 2px 10px rgba(225,29,72,0.4);
  }
  .cb-green {
    background: linear-gradient(135deg, #059669, #047857);
    color: white;
    box-shadow: 0 2px 10px rgba(5,150,105,0.4);
  }
 
  /* ── Live dot animation ── */
  @keyframes livepulse {
    0%, 100% { box-shadow: 0 0 6px #ef4444; opacity: 1; }
    50%       { box-shadow: 0 0 14px #ef4444; opacity: 0.7; }
  }
  .live-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: #ef4444;
    border-radius: 50%;
    animation: livepulse 1.4s ease-in-out infinite;
    vertical-align: middle;
    margin-right: 8px;
  }
 
  /* ── Captions (progress text, fps display) ── */
  .stCaption, [data-testid="stCaptionContainer"] p,
  div[data-testid="stCaptionContainer"] * {
    color: #ffd60a !important;
    opacity: 1 !important;
    font-weight: 600 !important;
  }
  /* General paragraph text in main area */
  .main .stMarkdown p {
    color: #ffffff !important;
  }
  /* Streamlit default text color — scoped only to markdown/custom content, NOT native widgets */
  .main .block-container .stMarkdown p,
  .main .block-container .stMarkdown span,
  .main .block-container .stMarkdown li,
  .main .block-container .stMarkdown h1,
  .main .block-container .stMarkdown h2,
  .main .block-container .stMarkdown h3 {
    color: #ffffff !important;
  }
 
  /* ── File uploader — force dark bg + readable text ── */
  [data-testid="stFileUploader"] section,
  [data-testid="stFileUploaderDropzone"],
  [data-testid="stFileUploader"] > div {
    background: #0d1117 !important;
    border-radius: 14px !important;
  }
  [data-testid="stFileUploaderDropzone"] *,
  [data-testid="stFileUploader"] span,
  [data-testid="stFileUploader"] p,
  [data-testid="stFileUploader"] small,
  [data-testid="stFileUploader"] button {
    color: #ffd60a !important;
  }
  [data-testid="stFileUploader"] button {
    background: rgba(255,107,53,0.15) !important;
    border: 1px solid #ff6b35 !important;
    color: #ff6b35 !important;
    border-radius: 8px !important;
  }
 
  /* ── Selectbox / dropdown — dark bg ── */
  [data-baseweb="select"] > div,
  [data-baseweb="popover"] ul,
  [data-baseweb="menu"] {
    background: #0d1117 !important;
    color: #ffffff !important;
  }
  [data-baseweb="select"] span,
  [data-baseweb="select"] div {
    color: #ffffff !important;
  }
</style>
""", unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════════════════════════════
# Sidebar – Settings panel
# ═══════════════════════════════════════════════════════════════════════
 
with st.sidebar:
 
    # ── Branded header ────────────────────────────────────────────────
    st.markdown("""
    <div style="
      background: linear-gradient(135deg, rgba(124,58,237,0.22), rgba(37,99,235,0.18));
      border: 1px solid rgba(124,58,237,0.35);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 18px;
      text-align: center;
    ">
      <div style="font-size:2rem; margin-bottom:4px;">🎯</div>
      <div style="font-size:1.15rem; font-weight:800; color:#a78bfa; letter-spacing:-0.01em;">Object Tracker</div>
      <div style="font-size:10px; color:#ffffff; margin-top:3px; letter-spacing:1.5px; text-transform:uppercase;">YOLOv8 · SORT · OpenCV</div>
    </div>
    """, unsafe_allow_html=True)
 
    # ── Model ────────────────────────────────────────────────────────
    st.markdown("""<div style="background:linear-gradient(90deg,rgba(124,58,237,0.18),rgba(37,99,235,0.08),transparent);
      border-left:3px solid #7c3aed; border-radius:0 8px 8px 0;
      padding:7px 12px; margin:0 0 10px 0;
      font-size:11px; font-weight:700; color:#a78bfa; letter-spacing:1.5px; text-transform:uppercase;">
      🧠 Detection Model</div>""", unsafe_allow_html=True)
 
    model_choice = st.selectbox(
        "YOLOv8 model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        index=0,
        help="Nano is fastest; Large is most accurate.",
    )
    model_labels = {
        "yolov8n.pt": "⚡ Nano — fastest (recommended for cloud)",
        "yolov8s.pt": "🚀 Small — balanced",
        "yolov8m.pt": "🎯 Medium — accurate, slow on CPU",
        "yolov8l.pt": "💎 Large — best accuracy, very slow on CPU",
    }
    st.caption(model_labels[model_choice])
 
    # Warn the user when picking a heavy model on Streamlit Cloud (CPU-only,
    # ~1 GB RAM). m / l will work but at ~0.5 FPS or less.
    if model_choice in ("yolov8m.pt", "yolov8l.pt"):
        st.warning(
            "⚠️ This model is heavy and runs on CPU on Streamlit Cloud "
            "(no GPU, limited RAM). Expect very slow processing. "
            "Use **yolov8n** or **yolov8s** for smooth performance.",
            icon="🐌",
        )
 
    conf_threshold = st.slider(
        "Confidence threshold", 0.1, 0.95, 0.25, 0.05,
        help="Minimum detection confidence to show a box.",
    )
    iou_threshold = st.slider(
        "NMS IOU threshold", 0.1, 0.95, 0.40, 0.05,
        help="Non-max suppression IOU for YOLO.",
    )
 
    st.markdown("---")
 
    # ── Tracker ──────────────────────────────────────────────────────
    st.markdown("""<div style="background:linear-gradient(90deg,rgba(37,99,235,0.18),rgba(8,145,178,0.08),transparent);
      border-left:3px solid #2563eb; border-radius:0 8px 8px 0;
      padding:7px 12px; margin:0 0 10px 0;
      font-size:11px; font-weight:700; color:#60a5fa; letter-spacing:1.5px; text-transform:uppercase;">
      🔁 Tracker</div>""", unsafe_allow_html=True)
 
    tracker_type = st.selectbox(
        "Algorithm",
        ["sort", "deepsort"] if WEBRTC_AVAILABLE else ["sort"],
        help="SORT is built-in. DeepSORT needs `pip install deep-sort-realtime`.",
    )
    max_age   = st.slider("Max age (frames)",  5, 100, 30, 5,
                          help="Frames to keep a track alive without a match.")
    min_hits  = st.slider("Min hits (SORT)",   1,  10,  3, 1,
                          help="Matched frames before a track is confirmed. "
                               "Higher = fewer flickering false tracks.")
    track_iou = st.slider("Tracker IOU",       0.1, 0.9, 0.3, 0.05,
                          help="Min IOU between detection and predicted track "
                               "for them to be matched.")
 
    st.markdown("---")
 
    # ── Class filter ─────────────────────────────────────────────────
    st.markdown("""<div style="background:linear-gradient(90deg,rgba(8,145,178,0.18),rgba(5,150,105,0.08),transparent);
      border-left:3px solid #0891b2; border-radius:0 8px 8px 0;
      padding:7px 12px; margin:0 0 10px 0;
      font-size:11px; font-weight:700; color:#34d399; letter-spacing:1.5px; text-transform:uppercase;">
      🏷️ Class Filter</div>""", unsafe_allow_html=True)
 
    COMMON_CLASSES = [
        "person", "car", "truck", "bus", "bicycle", "motorbike",
        "cat", "dog", "bottle", "chair", "laptop", "cell phone",
    ]
    selected_classes = st.multiselect(
        "Track only these classes",
        COMMON_CLASSES,
        default=[],
        help="Leave empty to track ALL 80 COCO classes.",
    )
    custom_class = st.text_input(
        "Add custom class", placeholder="e.g. aeroplane",
        help="Type any COCO class name not in the list above.",
    )
    if custom_class:
        selected_classes.append(custom_class.strip())
 
    target_classes = selected_classes if selected_classes else None
 
    st.markdown("---")
 
    # ── Display options ───────────────────────────────────────────────
    st.markdown("""<div style="background:linear-gradient(90deg,rgba(217,119,6,0.18),rgba(220,38,38,0.08),transparent);
      border-left:3px solid #d97706; border-radius:0 8px 8px 0;
      padding:7px 12px; margin:0 0 10px 0;
      font-size:11px; font-weight:700; color:#fbbf24; letter-spacing:1.5px; text-transform:uppercase;">
      🖥️ Display</div>""", unsafe_allow_html=True)
 
    show_fps    = st.toggle("Show FPS",           value=True)
    show_count  = st.toggle("Show object counts", value=True)
    frame_skip  = st.slider("Frame skip", 1, 5, 2, 1,
                            help="Process every Nth frame (higher = faster but choppy).")
    resize_width = st.selectbox(
        "Processing width (px)",
        [320, 480, 640, 854, 1280],
        index=2,
        help="Smaller = faster processing.",
    )
 
    st.markdown("---")
 
    # ── Footer ────────────────────────────────────────────────────────
    st.markdown("""
    <div style="
      text-align:center;
      background: linear-gradient(135deg, rgba(124,58,237,0.1), rgba(8,145,178,0.1));
      border: 1px solid rgba(124,58,237,0.15);
      border-radius: 12px;
      padding: 12px 10px;
    ">
      <div style="font-size:9px; color:#ffd60a; letter-spacing:2px; text-transform:uppercase; margin-bottom:5px;">Powered by</div>
      <div style="font-size:13px; font-weight:700;">
        <span style="color:#a78bfa;">YOLOv8</span>
        <span style="color:#ffd60a;"> · </span>
        <span style="color:#60a5fa;">SORT</span>
        <span style="color:#ffd60a;"> · </span>
        <span style="color:#34d399;">OpenCV</span>
        <span style="color:#ffd60a;"> · </span>
        <span style="color:#fbbf24;">Streamlit</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════════════════════════════
# Cached model loader  (reloads only when settings change)
# ═══════════════════════════════════════════════════════════════════════
 
# max_entries=1 ensures the previous detector is freed when the user
# switches model / changes settings — critical for Streamlit Cloud's
# limited (~1 GB) RAM. Without this, every change keeps an old copy
# of the YOLO weights in memory and the worker eventually OOMs (which
# is what made yolov8m / yolov8l "not work").
@st.cache_resource(show_spinner=False, max_entries=1)
def load_detector(model_path, conf, iou, classes, imgsz):
    return ObjectDetector(
        model_path=model_path,
        conf_threshold=conf,
        iou_threshold=iou,
        target_classes=classes if classes else None,
        imgsz=imgsz,
    )
 
 
# ═══════════════════════════════════════════════════════════════════════
# Helper – resize keeping aspect ratio
# ═══════════════════════════════════════════════════════════════════════
 
def resize_frame(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w == width:
        return frame
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)), interpolation=cv2.INTER_LINEAR)
 
 
# ═══════════════════════════════════════════════════════════════════════
# Helper – annotate a single frame
# ═══════════════════════════════════════════════════════════════════════
 
def annotate_frame(
    frame: np.ndarray,
    detector: ObjectDetector,
    tracker,
    fps_counter: FPSCounter,
    obj_counter: ObjectCounter,
    use_fps: bool = True,
    use_count: bool = True,
) -> tuple[np.ndarray, list[dict]]:
    """Run detection + tracking and draw results onto frame. Returns annotated frame + tracks."""
    detections = detector.detect(frame)
 
    if isinstance(tracker, SORTTracker):
        tracks = tracker.update(detections)
    else:
        tracks = tracker.update(detections, frame)
 
    fps_counter.tick()
    obj_counter.update(tracks, detector.class_names)
 
    annotated = frame.copy()
 
    for trk in tracks:
        label = format_label(
            detector.get_class_name(trk["class_id"]),
            trk["track_id"],
            trk["confidence"],
        )
        draw_track(annotated, trk, label)
 
    if use_fps:
        draw_fps(annotated, fps_counter.fps)
    if use_count:
        draw_object_count(annotated, obj_counter.counts)
 
    return annotated, tracks
 
 
# ═══════════════════════════════════════════════════════════════════════
# Helper – colorful class badges
# ═══════════════════════════════════════════════════════════════════════
 
_BADGE_COLORS = ["cb-purple", "cb-blue", "cb-teal", "cb-amber", "cb-rose", "cb-green"]
 
def make_badges(counts: dict) -> str:
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    html = ""
    for i, (cls, n) in enumerate(sorted_items):
        color = _BADGE_COLORS[i % len(_BADGE_COLORS)]
        html += f"<span class='class-badge {color}'>{cls} &nbsp;·&nbsp; {n}</span>"
    return html
 
 
# ═══════════════════════════════════════════════════════════════════════
# Main page – hero header
# ═══════════════════════════════════════════════════════════════════════
 
st.markdown("""
<div style="
  text-align: center;
  padding: 36px 20px 28px 20px;
  background:
    radial-gradient(ellipse at 30% 50%, rgba(124,58,237,0.12) 0%, transparent 60%),
    radial-gradient(ellipse at 70% 50%, rgba(37,99,235,0.10) 0%, transparent 60%),
    linear-gradient(135deg, rgba(124,58,237,0.06), rgba(37,99,235,0.06), rgba(8,145,178,0.06));
  border: 1px solid rgba(124,58,237,0.18);
  border-radius: 22px;
  margin-bottom: 24px;
">
  <div style="font-size:3.2rem; margin-bottom:10px; filter:drop-shadow(0 0 20px rgba(124,58,237,0.5));">🎯</div>
  <h1 style="
    background: linear-gradient(135deg, #a78bfa 0%, #60a5fa 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.7rem;
    font-weight: 900;
    margin: 0 0 10px 0;
    letter-spacing: -0.03em;
    line-height: 1.1;
  ">Object Detection & Tracking</h1>
  <p style="color:#ff6b35; font-size:1rem; margin:0 0 18px 0; letter-spacing:0.02em;">
    Real-time multi-class detection powered by YOLOv8 &nbsp;·&nbsp; SORT Tracker &nbsp;·&nbsp; OpenCV
  </p>
  <div style="display:flex; justify-content:center; gap:8px; flex-wrap:wrap;">
    <span style="background:rgba(124,58,237,0.15); color:#a78bfa; border:1px solid rgba(124,58,237,0.3);
      border-radius:99px; padding:4px 16px; font-size:11px; font-weight:700; letter-spacing:0.05em;">
      ⚡ REAL-TIME</span>
    <span style="background:rgba(37,99,235,0.15); color:#60a5fa; border:1px solid rgba(37,99,235,0.3);
      border-radius:99px; padding:4px 16px; font-size:11px; font-weight:700; letter-spacing:0.05em;">
      🧠 YOLOV8</span>
    <span style="background:rgba(8,145,178,0.15); color:#34d399; border:1px solid rgba(8,145,178,0.3);
      border-radius:99px; padding:4px 16px; font-size:11px; font-weight:700; letter-spacing:0.05em;">
      🎯 80 CLASSES</span>
    <span style="background:rgba(217,119,6,0.15); color:#fbbf24; border:1px solid rgba(217,119,6,0.3);
      border-radius:99px; padding:4px 16px; font-size:11px; font-weight:700; letter-spacing:0.05em;">
      📊 LIVE STATS</span>
  </div>
</div>
""", unsafe_allow_html=True)
 
 
# ═══════════════════════════════════════════════════════════════════════
# Tabs
# ═══════════════════════════════════════════════════════════════════════
 
tab_upload, tab_webcam, tab_about = st.tabs([
    "📁  Upload Video",
    "📷  Live Webcam",
    "ℹ️  About",
])
 
 
# ───────────────────────────────────────────────────────────────────────
# TAB 1 – Upload Video
# ───────────────────────────────────────────────────────────────────────
 
with tab_upload:
 
    st.markdown("""
    <div style="
      background: linear-gradient(90deg, rgba(124,58,237,0.08), transparent);
      border-left: 3px solid #7c3aed;
      border-radius: 0 10px 10px 0;
      padding: 10px 16px;
      margin-bottom: 16px;
    ">
      <span style="font-size:1.05rem; font-weight:700; color:#a78bfa;">📁 Upload a Video File</span>
      <span style="color:#ffffff; font-size:0.85rem; margin-left:10px;">MP4 · AVI · MOV · MKV · WEBM</span>
    </div>
    """, unsafe_allow_html=True)
 
    uploaded_file = st.file_uploader(
        "Drop your video here or click to browse",
        type=["mp4", "avi", "mov", "mkv", "webm"],
        label_visibility="collapsed",
    )
 
    if uploaded_file is not None:
 
        # ── Save upload to temp file ──────────────────────────────────
        suffix = Path(uploaded_file.name).suffix
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp_input.write(uploaded_file.read())
        tmp_input.flush()
        tmp_input.close()
 
        # ── Video info cards ──────────────────────────────────────────
        cap_info = cv2.VideoCapture(tmp_input.name)
        total_frames = int(cap_info.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps      = cap_info.get(cv2.CAP_PROP_FPS) or 30.0
        src_w        = int(cap_info.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h        = int(cap_info.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s   = total_frames / src_fps
        cap_info.release()
 
        # Colorful info cards
        st.markdown(f"""
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:16px 0;">
          <div style="background:linear-gradient(135deg,rgba(124,58,237,0.1),rgba(124,58,237,0.05));
            border:1px solid rgba(124,58,237,0.25); border-top:2px solid #7c3aed;
            border-radius:12px; padding:14px; text-align:center;">
            <div style="font-size:9px; color:#ff6b35; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Resolution</div>
            <div style="font-size:1.4rem; font-weight:800; color:#a78bfa; margin:4px 0;">{src_w}×{src_h}</div>
            <div style="font-size:10px; color:#ffffff;">pixels</div>
          </div>
          <div style="background:linear-gradient(135deg,rgba(37,99,235,0.1),rgba(37,99,235,0.05));
            border:1px solid rgba(37,99,235,0.25); border-top:2px solid #2563eb;
            border-radius:12px; padding:14px; text-align:center;">
            <div style="font-size:9px; color:#ff6b35; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Frames</div>
            <div style="font-size:1.4rem; font-weight:800; color:#60a5fa; margin:4px 0;">{total_frames:,}</div>
            <div style="font-size:10px; color:#ffffff;">total</div>
          </div>
          <div style="background:linear-gradient(135deg,rgba(8,145,178,0.1),rgba(8,145,178,0.05));
            border:1px solid rgba(8,145,178,0.25); border-top:2px solid #0891b2;
            border-radius:12px; padding:14px; text-align:center;">
            <div style="font-size:9px; color:#ff6b35; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">Duration</div>
            <div style="font-size:1.4rem; font-weight:800; color:#34d399; margin:4px 0;">{duration_s:.1f}s</div>
            <div style="font-size:10px; color:#ffffff;">seconds</div>
          </div>
          <div style="background:linear-gradient(135deg,rgba(217,119,6,0.1),rgba(217,119,6,0.05));
            border:1px solid rgba(217,119,6,0.25); border-top:2px solid #d97706;
            border-radius:12px; padding:14px; text-align:center;">
            <div style="font-size:9px; color:#ff6b35; text-transform:uppercase; letter-spacing:1.5px; font-weight:600;">FPS</div>
            <div style="font-size:1.4rem; font-weight:800; color:#fbbf24; margin:4px 0;">{src_fps:.0f}</div>
            <div style="font-size:10px; color:#ffffff;">source</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("---")
 
        st.caption(
            "ℹ️ Adjust sidebar settings (model / confidence / IOU / classes / "
            "tracker), **then** press **Start Processing**. Settings are "
            "captured at start time — changing sliders mid-run won't take "
            "effect until you start again."
        )
 
        # ── Start button ──────────────────────────────────────────────
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 2])
        with col_btn2:
            start = st.button("▶️  Start Processing", use_container_width=True)
 
        if start:
            # ── Load model ────────────────────────────────────────────
            with st.spinner(f"Loading {model_choice} …"):
                detector = load_detector(
                    model_choice, conf_threshold, iou_threshold,
                    tuple(target_classes) if target_classes else None,
                    int(resize_width),
                )
 
            tracker    = TrackerFactory.create(
                tracker_type,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=track_iou,
                reset_ids=True,   # restart IDs from 1 for each new run
            )
            fps_ctr    = FPSCounter(window=30)
            obj_ctr    = ObjectCounter()
 
            # ── Live Preview header ───────────────────────────────────
            st.markdown("""
            <div style="display:flex; align-items:center; gap:10px; margin:18px 0 10px 0;
              padding:10px 16px;
              background:linear-gradient(90deg,rgba(239,68,68,0.08),transparent);
              border-left:3px solid #ef4444; border-radius:0 10px 10px 0;">
              <span class="live-dot"></span>
              <span style="font-size:1rem; font-weight:700; color:#fca5a5; letter-spacing:0.05em;">LIVE PREVIEW</span>
            </div>
            """, unsafe_allow_html=True)
 
            frame_placeholder = st.empty()
 
            # ── Progress row ──────────────────────────────────────────
            prog_col, stat_col = st.columns([3, 1])
            with prog_col:
                progress_bar  = st.progress(0.0)
                progress_text = st.empty()
            with stat_col:
                fps_display   = st.empty()
 
            # ── Stats header ──────────────────────────────────────────
            st.markdown("""
            <div style="
              background:linear-gradient(90deg,rgba(124,58,237,0.1),rgba(37,99,235,0.08),transparent);
              border-left:3px solid #7c3aed; border-radius:0 10px 10px 0;
              padding:9px 16px; margin:16px 0 10px 0;
              font-size:11px; font-weight:700; color:#a78bfa; letter-spacing:1.5px; text-transform:uppercase;">
              📊 Live Statistics
            </div>
            """, unsafe_allow_html=True)
 
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            metric_fps   = stat_col1.empty()
            metric_objs  = stat_col2.empty()
            metric_frame = stat_col3.empty()
            metric_ids   = stat_col4.empty()
 
            count_display = st.empty()
            def _set_stop(): st.session_state["_stop_upload"] = True
            stop_btn      = st.button("⏹️  Stop", key="stop_upload", on_click=_set_stop)
 
            # ── Prepare output writer ─────────────────────────────────
            # IMPORTANT: when frame_skip > 1 we only write every Nth frame,
            # so the output FPS must be source_fps / frame_skip — otherwise
            # the saved video plays back at N× speed and looks choppy.
            proc_w       = resize_width
            proc_h       = int(src_h * proc_w / src_w)
            output_fps   = max(1.0, src_fps / max(1, frame_skip))
            tmp_output   = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_output.close()
            fourcc       = cv2.VideoWriter_fourcc(*"mp4v")
            writer       = cv2.VideoWriter(tmp_output.name, fourcc, output_fps, (proc_w, proc_h))
 
            # ── Processing loop ───────────────────────────────────────
            cap        = cv2.VideoCapture(tmp_input.name)
            frame_idx  = 0
            all_ids: set[int] = set()
            st.session_state["_stop_upload"] = False
 
            while cap.isOpened():
                if st.session_state.get("_stop_upload", False):
                    break
 
                ok, frame = cap.read()
                if not ok:
                    break
 
                frame_idx += 1
 
                if frame_idx % frame_skip != 0:
                    continue
 
                frame = resize_frame(frame, proc_w)
 
                annotated, tracks = annotate_frame(
                    frame, detector, tracker,
                    fps_ctr, obj_ctr, show_fps, show_count,
                )
 
                all_ids.update(t["track_id"] for t in tracks)
                writer.write(annotated)
 
                # Show every processed frame in the live preview
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb, channels="RGB", use_container_width=True)
 
                pct = (frame_idx / total_frames) if total_frames > 0 else 0
                progress_bar.progress(min(pct, 1.0))
                progress_text.caption(
                    f"Frame {frame_idx:,} / {total_frames:,}  ({pct*100:.1f}%)"
                )
 
                fps_val = fps_ctr.fps
                fps_display.caption(f"⚡ {fps_val:.1f} FPS")
 
                metric_fps.metric("⚡ FPS",           f"{fps_val:.1f}")
                metric_objs.metric("🎯 Active tracks", len(tracks))
                metric_frame.metric("🎞️ Frame",        frame_idx)
                metric_ids.metric("🆔 Unique IDs",    len(all_ids))
 
                counts = obj_ctr.counts
                if counts:
                    count_display.markdown(
                        "<div style='margin-top:8px;'>" + make_badges(counts) + "</div>",
                        unsafe_allow_html=True
                    )
 
            cap.release()
            writer.release()
 
            progress_bar.progress(1.0)
            progress_text.caption("✅ Processing complete!")
 
            # ── Success banner ────────────────────────────────────────
            st.markdown(f"""
            <div style="
              background: linear-gradient(135deg, rgba(5,150,105,0.12), rgba(8,145,178,0.10));
              border: 1px solid rgba(52,211,153,0.3);
              border-left: 3px solid #34d399;
              border-radius: 12px;
              padding: 14px 20px;
              margin: 16px 0;
              display: flex; align-items: center; gap: 12px;
            ">
              <span style="font-size:1.5rem;">✅</span>
              <div>
                <div style="font-weight:700; color:#34d399; font-size:0.95rem;">Processing Complete!</div>
                <div style="color:#ff6b35; font-size:0.82rem; margin-top:2px;">
                  {frame_idx:,} frames processed &nbsp;·&nbsp; {len(all_ids)} unique object IDs tracked
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
 
            # ── Download button ───────────────────────────────────────
            with open(tmp_output.name, "rb") as f:
                video_bytes = f.read()
 
            st.markdown("""
            <div style="
              background:linear-gradient(90deg,rgba(8,145,178,0.1),transparent);
              border-left:3px solid #0891b2; border-radius:0 10px 10px 0;
              padding:9px 16px; margin:16px 0 10px 0;
              font-size:11px; font-weight:700; color:#34d399; letter-spacing:1.5px; text-transform:uppercase;">
              💾 Download Processed Video
            </div>
            """, unsafe_allow_html=True)
 
            dl_col1, dl_col2, dl_col3 = st.columns([2, 1, 2])
            with dl_col2:
                st.download_button(
                    label="⬇️  Download MP4",
                    data=video_bytes,
                    file_name=f"tracked_{uploaded_file.name}",
                    mime="video/mp4",
                    use_container_width=True,
                )
 
            try:
                os.unlink(tmp_input.name)
                os.unlink(tmp_output.name)
            except Exception:
                pass
 
    else:
        # ── Empty state ───────────────────────────────────────────────
        st.markdown("""
        <div style="
          text-align: center;
          padding: 72px 24px;
          background:
            radial-gradient(ellipse at 50% 50%, rgba(124,58,237,0.07) 0%, transparent 70%),
            linear-gradient(135deg, rgba(124,58,237,0.04), rgba(37,99,235,0.04), rgba(8,145,178,0.04));
          border: 2px dashed rgba(124,58,237,0.22);
          border-radius: 22px;
          margin-top: 24px;
        ">
          <div style="font-size:4rem; margin-bottom:14px; filter:drop-shadow(0 0 18px rgba(124,58,237,0.4));">🎬</div>
          <h3 style="
            background: linear-gradient(135deg, #a78bfa, #60a5fa);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin: 0 0 10px 0; font-size:1.5rem; font-weight:800;
          ">Drop your video to begin</h3>
          <p style="color:#ffffff; margin:0 0 22px 0; font-size:0.9rem;">
            Upload a file using the uploader above, then click <strong style="color:#a78bfa;">Start Processing</strong>
          </p>
          <div style="display:flex; justify-content:center; gap:8px; flex-wrap:wrap;">
            <span style="background:rgba(124,58,237,0.12); color:#a78bfa; border:1px solid rgba(124,58,237,0.25);
              border-radius:99px; padding:5px 16px; font-size:11px; font-weight:700;">.MP4</span>
            <span style="background:rgba(37,99,235,0.12); color:#60a5fa; border:1px solid rgba(37,99,235,0.25);
              border-radius:99px; padding:5px 16px; font-size:11px; font-weight:700;">.AVI</span>
            <span style="background:rgba(8,145,178,0.12); color:#34d399; border:1px solid rgba(8,145,178,0.25);
              border-radius:99px; padding:5px 16px; font-size:11px; font-weight:700;">.MOV</span>
            <span style="background:rgba(217,119,6,0.12); color:#fbbf24; border:1px solid rgba(217,119,6,0.25);
              border-radius:99px; padding:5px 16px; font-size:11px; font-weight:700;">.MKV</span>
            <span style="background:rgba(220,38,38,0.12); color:#f87171; border:1px solid rgba(220,38,38,0.25);
              border-radius:99px; padding:5px 16px; font-size:11px; font-weight:700;">.WEBM</span>
          </div>
        </div>
        """, unsafe_allow_html=True)
 
 
# ───────────────────────────────────────────────────────────────────────
# TAB 2 – Live Webcam
# ───────────────────────────────────────────────────────────────────────
 
with tab_webcam:
 
    st.markdown("""
    <div style="
      background: linear-gradient(90deg, rgba(37,99,235,0.1), transparent);
      border-left: 3px solid #2563eb;
      border-radius: 0 10px 10px 0;
      padding: 10px 16px;
      margin-bottom: 16px;
    ">
      <span style="font-size:1.05rem; font-weight:700; color:#60a5fa;">📷 Live Webcam Tracking</span>
    </div>
    """, unsafe_allow_html=True)
 
    if WEBRTC_AVAILABLE:
        st.info(
            "🟢 **WebRTC mode** — click **START** to begin live detection. "
            "Allow browser camera access when prompted.",
            icon="📷",
        )
 
        st.session_state["wb_model"]    = model_choice
        st.session_state["wb_conf"]     = conf_threshold
        st.session_state["wb_iou"]      = iou_threshold
        st.session_state["wb_classes"]  = tuple(target_classes) if target_classes else None
        st.session_state["wb_tracker"]  = tracker_type
        st.session_state["wb_max_age"]  = max_age
        st.session_state["wb_min_hits"] = min_hits
        st.session_state["wb_track_iou"]= track_iou
        st.session_state["wb_fps"]      = show_fps
        st.session_state["wb_count"]    = show_count
        st.session_state["wb_width"]    = resize_width
 
        st.caption(
            "ℹ️ Webcam settings are baked in when **START** is pressed. "
            "Changing sliders mid-stream has no effect — press **STOP** "
            "and **START** again to apply new settings."
        )
 
        class LiveVideoProcessor(VideoProcessorBase):
            def __init__(self):
                ss = st.session_state
                self.detector = ObjectDetector(
                    model_path=ss.get("wb_model", "yolov8n.pt"),
                    conf_threshold=ss.get("wb_conf", 0.25),
                    iou_threshold=ss.get("wb_iou", 0.40),
                    target_classes=ss.get("wb_classes", None),
                    imgsz=int(ss.get("wb_width", 640)),
                )
                self.tracker = TrackerFactory.create(
                    ss.get("wb_tracker", "sort"),
                    max_age=ss.get("wb_max_age", 30),
                    min_hits=ss.get("wb_min_hits", 3),
                    iou_threshold=ss.get("wb_track_iou", 0.3),
                    reset_ids=True,
                )
                self.fps_ctr  = FPSCounter(window=30)
                self.obj_ctr  = ObjectCounter()
                self.width    = ss.get("wb_width", 640)
                self.use_fps  = ss.get("wb_fps", True)
                self.use_cnt  = ss.get("wb_count", True)
 
            def recv(self, frame: "av.VideoFrame") -> "av.VideoFrame":
                img = frame.to_ndarray(format="bgr24")
                img = resize_frame(img, self.width)
                annotated, _ = annotate_frame(
                    img, self.detector, self.tracker,
                    self.fps_ctr, self.obj_ctr,
                    self.use_fps, self.use_cnt,
                )
                return av.VideoFrame.from_ndarray(annotated, format="bgr24")
 
        RTC_CONFIG = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )
        webrtc_streamer(
            key="live-tracker",
            video_processor_factory=LiveVideoProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
 
    else:
        st.warning(
            "⚠️ `streamlit-webrtc` is not installed. "
            "Falling back to local OpenCV webcam mode.  \n"
            "For WebRTC support: `pip install streamlit-webrtc av`",
            icon="⚠️",
        )
 
        cam_col1, cam_col2, cam_col3 = st.columns([2, 1, 2])
        with cam_col2:
            start_cam = st.button("▶️  Start Webcam", use_container_width=True)
 
        if start_cam:
            with st.spinner(f"Loading {model_choice} …"):
                detector = load_detector(
                    model_choice, conf_threshold, iou_threshold,
                    tuple(target_classes) if target_classes else None,
                    int(resize_width),
                )
 
            tracker = TrackerFactory.create(
                tracker_type,
                max_age=max_age,
                min_hits=min_hits,
                iou_threshold=track_iou,
                reset_ids=True,
            )
            fps_ctr = FPSCounter(window=30)
            obj_ctr = ObjectCounter()
 
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open webcam. Make sure a camera is connected.")
                st.stop()
 
            st.markdown("""
            <div style="display:flex; align-items:center; gap:10px; margin:14px 0 8px 0;
              padding:10px 16px;
              background:linear-gradient(90deg,rgba(239,68,68,0.08),transparent);
              border-left:3px solid #ef4444; border-radius:0 10px 10px 0;">
              <span class="live-dot"></span>
              <span style="font-size:1rem; font-weight:700; color:#fca5a5; letter-spacing:0.05em;">LIVE FEED</span>
            </div>
            """, unsafe_allow_html=True)
 
            frame_slot  = st.empty()
            count_slot  = st.empty()
            stop_cam    = st.button("⏹️  Stop Webcam", key="stop_cam")
 
            while cap.isOpened():
                if stop_cam:
                    break
 
                ok, frame = cap.read()
                if not ok:
                    st.warning("⚠️ Lost camera feed.")
                    break
 
                frame = resize_frame(frame, resize_width)
                annotated, tracks = annotate_frame(
                    frame, detector, tracker,
                    fps_ctr, obj_ctr, show_fps, show_count,
                )
 
                rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                frame_slot.image(rgb, channels="RGB", use_container_width=True)
 
                counts = obj_ctr.counts
                if counts:
                    count_slot.markdown(
                        "<div style='margin-top:8px;'>" + make_badges(counts) + "</div>",
                        unsafe_allow_html=True
                    )
 
            cap.release()
            st.info("📷 Webcam stopped.")
 
 
# ───────────────────────────────────────────────────────────────────────
# TAB 3 – About
# ───────────────────────────────────────────────────────────────────────
 
with tab_about:
 
    st.markdown("""
    <div style="
      background: linear-gradient(90deg, rgba(8,145,178,0.1), transparent);
      border-left: 3px solid #0891b2;
      border-radius: 0 10px 10px 0;
      padding: 10px 16px;
      margin-bottom: 20px;
    ">
      <span style="font-size:1.05rem; font-weight:700; color:#34d399;">ℹ️ About This App</span>
    </div>
    """, unsafe_allow_html=True)
 
    col_a, col_b = st.columns(2)
 
    with col_a:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(124,58,237,0.07),rgba(37,99,235,0.05));
          border:1px solid rgba(124,58,237,0.18); border-radius:14px; padding:20px; margin-bottom:14px;">
          <div style="font-size:11px; color:#ff6b35; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:12px;">🔧 Technology Stack</div>
        """, unsafe_allow_html=True)
 
        st.markdown("""
        | Component | Library |
        |---|---|
        | Detection | YOLOv8 (Ultralytics) |
        | Tracking | SORT (Kalman + Hungarian) |
        | Vision | OpenCV |
        | Web App | Streamlit |
        | Webcam | streamlit-webrtc |
        """)
 
        st.markdown("</div>", unsafe_allow_html=True)
 
        st.markdown("""
        #### 📐 Architecture
        ```
        Frame
          ↓
        ObjectDetector (YOLOv8)
          ↓ [x1,y1,x2,y2, conf, class_id]
        SORTTracker (Kalman filter)
          ↓ [track_id, bbox, conf, class_id]
        draw_track() → annotated frame
          ↓
        st.image() / WebRTC stream
        ```
        """)
 
    with col_b:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(8,145,178,0.07),rgba(5,150,105,0.05));
          border:1px solid rgba(8,145,178,0.18); border-radius:14px; padding:20px; margin-bottom:14px;">
          <div style="font-size:11px; color:#ff6b35; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:12px;">🎮 Controls</div>
          <div style="color:#ffffff; font-size:0.88rem; line-height:1.8;">
            <strong style="color:#60a5fa;">Sidebar</strong> — adjust all settings before starting:<br>
            &nbsp;· Model size (speed vs accuracy)<br>
            &nbsp;· Confidence &amp; IOU thresholds<br>
            &nbsp;· Class filter (person, car, etc.)<br>
            &nbsp;· Tracker settings<br>
            &nbsp;· Display options<br><br>
            <strong style="color:#34d399;">During processing:</strong><br>
            &nbsp;· ⏹️ Stop button halts processing<br>
            &nbsp;· ⬇️ Download processed video after completion
          </div>
        </div>
        """, unsafe_allow_html=True)
 
        st.markdown("""
        #### 📦 Project Files
        ```
        streamlit.py      ← this app
        main.py           ← CLI entry point
        detector.py       ← YOLOv8 wrapper
        tracker.py        ← SORT / DeepSORT
        utils.py          ← drawing helpers
        ```
        """)
 
    st.markdown("---")
 
    st.markdown("""
    <div style="display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-bottom:20px;">
      <div style="background:linear-gradient(135deg,rgba(217,119,6,0.08),rgba(217,119,6,0.04));
        border:1px solid rgba(217,119,6,0.2); border-radius:14px; padding:18px;">
        <div style="font-size:11px; color:#ff6b35; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:10px;">🚀 Quick Setup</div>
        <code style="color:#fbbf24; font-size:0.82rem;">pip install -r requirements.txt</code><br>
        <code style="color:#fbbf24; font-size:0.82rem;">streamlit run streamlit.py</code>
      </div>
      <div style="background:linear-gradient(135deg,rgba(5,150,105,0.08),rgba(8,145,178,0.04));
        border:1px solid rgba(5,150,105,0.2); border-radius:14px; padding:18px;">
        <div style="font-size:11px; color:#ff6b35; text-transform:uppercase; letter-spacing:1.5px; font-weight:600; margin-bottom:10px;">🌐 Cloud-Ready</div>
        <div style="color:#ffffff; font-size:0.82rem; line-height:1.8;">
          Optimized for fast deployment with<br>
          minimal setup. Works out of the box on<br>
          most cloud platforms. ✨
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
 
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:14px 0 4px 0;">
      <span style="font-size:13px; color:#ffd60a;">Built with ❤️ using &nbsp;</span>
      <span style="color:#a78bfa; font-weight:700;">YOLOv8</span>
      <span style="color:#ffd60a;"> · </span>
      <span style="color:#60a5fa; font-weight:700;">SORT</span>
      <span style="color:#ffd60a;"> · </span>
      <span style="color:#34d399; font-weight:700;">OpenCV</span>
      <span style="color:#ffd60a;"> · </span>
      <span style="color:#fbbf24; font-weight:700;">Streamlit</span>
    </div>
    """, unsafe_allow_html=True)