
import io
import requests
from typing import Tuple, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import cv2
import matplotlib.pyplot as plt

st.set_page_config(page_title="Image Processing Studio", page_icon="🖼️", layout="wide")

st.title("🖼️ Image Processing Studio (Streamlit)")
st.caption("แหล่งภาพ: อัปโหลด / กล้องเว็บแคม / URL  |  ปรับพารามิเตอร์แบบเรียลไทม์  |  กราฟคุณสมบัติของรูปภาพ")

# ---------------------- Utilities ----------------------
def pil_to_cv(img_pil: Image.Image) -> np.ndarray:
    """PIL (RGB) -> OpenCV (BGR)"""
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def cv_to_pil(img_cv: np.ndarray) -> Image.Image:
    """OpenCV (BGR or Gray) -> PIL (RGB)"""
    if len(img_cv.shape) == 2:
        return Image.fromarray(img_cv)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def read_image_from_url(url: str) -> Optional[Image.Image]:
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        st.error(f"โหลดรูปจาก URL ไม่ได้: {e}")
        return None

def ensure_max_size(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    scale = max(w, h)
    if scale <= max_side:
        return img
    ratio = max_side / scale
    return img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

# ---------------------- Sidebar: Source ----------------------
st.sidebar.header("1) เลือกแหล่งที่มาของรูปภาพ")
source = st.sidebar.radio(
    "Source",
    ["Upload", "Webcam", "URL"],
    captions=["อัปโหลดไฟล์รูป", "ถ่ายภาพผ่านกล้องโน้ตบุ๊ค/เว็บแคม", "ดึงรูปจากอินเทอร์เน็ต"]
)

img_pil = None

if source == "Upload":
    file = st.sidebar.file_uploader("อัปโหลดรูปภาพ (PNG/JPG)", type=["png", "jpg", "jpeg", "bmp", "webp"])
    if file is not None:
        img_pil = Image.open(file).convert("RGB")
elif source == "Webcam":
    snap = st.sidebar.camera_input("ถ่ายภาพจากกล้อง")
    if snap is not None:
        img_pil = Image.open(snap).convert("RGB")
elif source == "URL":
    default_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/2/24/WelshCorgi.jpeg/640px-WelshCorgi.jpeg"
    url = st.sidebar.text_input("วางลิงก์รูปภาพ (URL)", value=default_url)
    if url:
        img_pil = read_image_from_url(url)

if img_pil is None:
    st.info("⬅️ เลือกและโหลดรูปภาพจากแถบด้านซ้ายก่อน")
    st.stop()

img_pil = ensure_max_size(img_pil, max_side=1400)
img_cv = pil_to_cv(img_pil)  # BGR

# ---------------------- Sidebar: Processing ----------------------
st.sidebar.header("2) เลือกการประมวลผลภาพ")
op = st.sidebar.selectbox(
    "ชนิดของ Image Processing",
    ["None", "Grayscale", "Gaussian Blur", "Canny Edge", "Brightness/Contrast", "Threshold (Binary)", "Sharpen"]
)

# parameter widgets
params = {}
if op == "Gaussian Blur":
    params["kernel"] = st.sidebar.slider("ขนาด Kernel (คี่)", min_value=1, max_value=51, value=9, step=2)
if op == "Canny Edge":
    params["thr1"] = st.sidebar.slider("Threshold 1", 0, 500, 100, 5)
    params["thr2"] = st.sidebar.slider("Threshold 2", 0, 500, 200, 5)
if op == "Brightness/Contrast":
    params["alpha"] = st.sidebar.slider("ความคมชัด (α)", 0.10, 3.00, 1.20, 0.05)
    params["beta"] = st.sidebar.slider("ความสว่าง (β)", -100, 100, 10, 1)
if op == "Threshold (Binary)":
    params["thresh"] = st.sidebar.slider("Threshold", 0, 255, 127, 1)
if op == "Sharpen":
    params["amount"] = st.sidebar.slider("ความคมชัด", 0.0, 3.0, 1.0, 0.05)

# ---------------------- Processing ----------------------
processed = img_cv.copy()
aux_for_graph = None  # grayscale used for histogram

if op == "None":
    processed = img_cv

elif op == "Grayscale":
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    processed = gray
    aux_for_graph = gray

elif op == "Gaussian Blur":
    k = params["kernel"]
    processed = cv2.GaussianBlur(img_cv, (k, k), 0)

elif op == "Canny Edge":
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, params["thr1"], params["thr2"])
    processed = edges
    aux_for_graph = gray

elif op == "Brightness/Contrast":
    a = params["alpha"]
    b = params["beta"]
    processed = cv2.convertScaleAbs(img_cv, alpha=a, beta=b)

elif op == "Threshold (Binary)":
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, params["thresh"], 255, cv2.THRESH_BINARY)
    processed = bw
    aux_for_graph = gray

elif op == "Sharpen":
    amt = params["amount"]
    # Unsharp masking
    blur = cv2.GaussianBlur(img_cv, (0, 0), sigmaX=3)
    sharpen = cv2.addWeighted(img_cv, 1 + amt, blur, -amt, 0)
    processed = sharpen

# ---------------------- Display Images ----------------------
col1, col2 = st.columns(2, gap="large")
with col1:
    st.subheader("ภาพต้นฉบับ")
    st.image(img_pil, use_column_width=True)

with col2:
    st.subheader("ผลลัพธ์หลังประมวลผล")
    st.image(cv_to_pil(processed), use_column_width=True)

# ---------------------- Metrics & Graph ----------------------
st.markdown("---")
st.header("📈 กราฟคุณสมบัติของรูปภาพ")

h, w = processed.shape[:2]
channels = 1 if len(processed.shape) == 2 else processed.shape[2]

# คำนวณ grayscale สำหรับ metric
if aux_for_graph is None:
    gray_all = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY) if channels == 3 else processed
else:
    gray_all = aux_for_graph

mean_val = float(np.mean(gray_all))
std_val = float(np.std(gray_all))

edge_pixels = int(np.count_nonzero(processed)) if processed.ndim == 2 and processed.max() in (1, 255) else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("กว้าง (px)", w)
m2.metric("สูง (px)", h)
m3.metric("Mean Gray", f"{mean_val:.2f}")
m4.metric("Std Gray", f"{std_val:.2f}")

if edge_pixels is not None:
    st.info(f"จำนวนพิกเซลที่เป็นเส้นขอบ (Edges): {edge_pixels:,} px")

# ---------------------- Graph Options ----------------------
st.subheader("เลือกชนิดกราฟ Histogram")
graph_type = st.radio(
    "ชนิดของกราฟ",
    ["Gray Histogram", "RGB Histogram"],
    horizontal=True
)

if graph_type == "Gray Histogram":
    fig = plt.figure(figsize=(6, 4))
    plt.hist(gray_all.ravel(), bins=32, range=(0, 255), color="gray")
    plt.title("Histogram of gray level intensity (0–255)")
    plt.xlabel("Intensity value")
    plt.ylabel("Number of pixels")
    st.pyplot(fig)

elif graph_type == "RGB Histogram":
    fig = plt.figure(figsize=(6, 4))
    color_map = {"b": "blue", "g": "green", "r": "red"}
    for i, col in enumerate(("b", "g", "r")):
        hist = cv2.calcHist([processed], [i], None, [32], [0, 256])
        plt.plot(hist, color=color_map[col])
        plt.xlim([0, 32])
    plt.title("Histogram of RGB channels (R, G, B)")
    plt.xlabel("Bins")
    plt.ylabel("Number of pixels")
    st.pyplot(fig)

# ---------------------- Save Processed Image ----------------------
st.subheader("💾 ดาวน์โหลดผลลัพธ์")
buf = io.BytesIO()
cv_to_pil(processed).save(buf, format="PNG")
st.download_button(
    label="ดาวน์โหลดรูปหลังประมวลผล (PNG)",
    data=buf.getvalue(),
    file_name="processed_image.png",
    mime="image/png"
)