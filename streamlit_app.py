import streamlit as st
import os
from ultralytics import YOLO
import easyocr
import glob
import cv2
import shutil
import torch
import requests

# -----------------------------
# Config
# -----------------------------
MODEL_URL = "https://huggingface.co/your-username/carplate-ocr/resolve/main/best.pt"
MODEL_PATH = "best.pt"

# -----------------------------
# Download model if not exists
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading YOLOv8 model weights... please wait ⏳")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    st.success("Model downloaded successfully!")

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🚗 Car Plate License Detection & OCR")

uploaded_file = st.file_uploader("Upload a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    input_path = "input.jpg"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    # Clear old cropped results
    if os.path.exists("cropped_plates"):
        shutil.rmtree("cropped_plates")

    # Load YOLOv8 model
    model = YOLO(MODEL_PATH)

    # Run inference
    results = model(input_path)
    results[0].save_crop("cropped_plates")

    # Show detection result
    st.image(input_path, caption="Detected License Plate", use_column_width=True)

    # OCR
    reader = easyocr.Reader(['en'])
    image_paths = glob.glob("cropped_plates/**/*.jpg", recursive=True) + \
                  glob.glob("cropped_plates/**/*.png", recursive=True)

    if not image_paths:
        st.error("❌ No license plate detected.")
    else:
        for path in image_paths:
            img = cv2.imread(path)
            result = reader.readtext(img)
            if result:
                st.success(f"[{os.path.basename(path)}] Text: {result[0][1]}")
            else:
                st.warning(f"[{os.path.basename(path)}] No text detected")
