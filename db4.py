import streamlit as st
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import re
import av
import os
from ultralytics import YOLO
import yt_dlp
import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

# --- Caching Resources ---

@st.cache_resource
def load_model():
    return YOLO("crowdyolov8n.onnx", task="detect")

@st.cache_resource
def load_tracker():
    return DeepSort(max_age=30)

@st.cache_data(show_spinner=False)
def download_youtube_video_cached(url):
    output_path = "temp_video.mp4"
    ydl_opts = {
        "format": "bestvideo[ext=mp4]",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return output_path
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def fetch_image_bytes_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    return response.content

# Load model & tracker once
model = load_model()
tracker = load_tracker()

# --- Custom Styles ---
st.markdown("""
<style>
    .stApp {
        margin-top: 50px;
        background: linear-gradient(to bottom right, #8e44ad, #3498db, #2ecc71);
    }
    .neon-title {
        width: 100%;
        text-align: center;
        margin-top: 45px;
        padding: 20px;
        border-radius: 12px;
        background: transparent;
        border: 3px solid #00ffff;
        box-shadow: 0px 0px 20px #00ffff;
        animation: neonGlow 1.5s infinite alternate;
        font-size: 45px;
        font-weight: bold;
        color: white;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stButton > button {
        width: 50%;
        text-align: center;
        background: white !important;
        color: black !important;
        border-radius: 6px !important;
        font-weight: bold;
        padding: 10px;
        margin-top:20px;
        margin-left:180px;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "page" not in st.session_state:
    st.session_state.page = "upload"
if "media" not in st.session_state:
    st.session_state.media = None
if "media_type" not in st.session_state:
    st.session_state.media_type = None

def reset_app_state():
    for key in ["page", "media", "media_type", "processed_video_frames", "video_people_count"]:
        if key in st.session_state:
            del st.session_state[key]
    st.session_state.page = "upload"
    #st.experimental_rerun()


st.markdown('<div class="neon-title">CROWD DETECTION</div>', unsafe_allow_html=True)

# --- Helpers ---
def extract_youtube_id(url):
    match = re.search(r"(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&=%\?]{11})", url)
    return match.group(1) if match else None

def fetch_media_from_url(url):
    try:
        youtube_id = extract_youtube_id(url)
        if youtube_id:
            st.info("Downloading and processing YouTube video...")
            video_path = download_youtube_video_cached(url)
            if video_path and os.path.exists(video_path):
                st.session_state.media = video_path
                st.session_state.media_type = "video"
                st.session_state.page = "preview"
                st.rerun()
                return

        if any(url.lower().endswith(ext) for ext in ["jpg", "jpeg", "png"]):
            img_bytes = fetch_image_bytes_from_url(url)
            img = Image.open(BytesIO(img_bytes)).convert("RGB")
            st.session_state.media = img
            st.session_state.media_type = "image"
        elif any(url.lower().endswith(ext) for ext in ["mp4", "mov", "avi"]):
            st.session_state.media = url
            st.session_state.media_type = "video"
        else:
            st.error("Unsupported URL media format.")
            return

        st.session_state.page = "preview"
        st.rerun()

    except Exception as e:
        st.error(f"Failed to fetch media: {e}")

# --- Pages ---
def show_upload_page():
    file = st.file_uploader("Upload Image or Video", type=["jpg", "png", "jpeg", "mp4", "mov", "avi"])
    media_url = st.text_input("Enter Image, Video, or YouTube URL")

    if file:
        if file.type.startswith("image"):
            img = Image.open(file).convert("RGB")
            st.session_state.media = img
            st.session_state.media_type = "image"
        else:
            st.session_state.media = file
            st.session_state.media_type = "video"
        st.session_state.page = "preview"
        st.rerun()
    elif media_url:
        fetch_media_from_url(media_url)

def show_image_preview():
    image = st.session_state.media
    results = model(image, conf=0.2, iou=0.5)[0]
    draw = ImageDraw.Draw(image)
    boxes = results.boxes.xyxy.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy().astype(int)

    people_count = 0
    for i, box in enumerate(boxes):
        if classes[i] == 0:
            people_count += 1
            draw.rectangle(box, outline="red", width=3)

    st.image(image, use_container_width=True)
    st.markdown(f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: white;'>👥 People Count: <span style='color: #00ffff;'>{people_count}</span></div>", unsafe_allow_html=True)

    if st.button("Try Another Image or Video"):
        reset_app_state()

def show_video_preview():
    stframe = st.empty()
    count_placeholder = st.empty()

    # --- Run only once ---
    if "processed_video_frames" not in st.session_state:
        video = av.open(st.session_state.media)
        frames = []
        max_people_detected = 0

        for frame in video.decode(video=0):
            img = frame.to_ndarray(format="bgr24")

            results = model(img, conf=0.2, iou=0.5)[0]
            people_in_frame = sum(1 for cls in results.boxes.cls if int(cls.item()) == 0)
            max_people_detected = max(max_people_detected, people_in_frame)

            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if int(cls.item()) == 0:
                    x1, y1, x2, y2 = map(int, box.tolist())
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Store results in session_state
        st.session_state.processed_video_frames = frames
        st.session_state.video_people_count = max_people_detected

    # --- Display video ---
    for frame in st.session_state.processed_video_frames:
        stframe.image(frame, use_container_width=True)

    # --- Show count ---
    count_placeholder.markdown(
        f"<div style='text-align: center; font-size: 24px; font-weight: bold; color: white;'>👥 Estimated Total People in Video: <span style='color: #00ffff;'>{st.session_state.video_people_count}</span></div>",
        unsafe_allow_html=True
    )

    # --- Reset Button ---
    if st.button("Try Another Image or Video"):
        reset_app_state()




# --- App Routing ---
if st.session_state.page == "upload":
    show_upload_page()
elif st.session_state.page == "preview":
    if st.session_state.media_type == "image":
        show_image_preview()
    elif st.session_state.media_type == "video":
        show_video_preview()
