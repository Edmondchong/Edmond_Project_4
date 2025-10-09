import streamlit as st
import os
import requests
from ultralytics import YOLO
import easyocr
import glob
import cv2
import shutil

# -------------------------------
# Hugging Face Model Settings
# -------------------------------
MODEL_URL = "https://huggingface.co/EdmondChong/License_Plate_Recognition/resolve/main/best.pt"
MODEL_PATH = "best.pt"
HF_TOKEN = st.secrets["HF_TOKEN"]  # stored securely in Streamlit Cloud

# -------------------------------
# Download model if not exists
# -------------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading YOLOv8 model weights from Hugging Face...")
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        r = requests.get(MODEL_URL, headers=headers, stream=True)
        r.raise_for_status()
        with open(MODEL_PATH, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        st.success("Model downloaded!")

    # Load YOLOv8 model
    model = YOLO(MODEL_PATH)
    return model

model = load_model()

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🚗 Edmond Chong's Car Plate Recognition App")
st.write("Upload a car image and click **Detect** to recognize the license plate.")

uploaded_file = st.file_uploader("Upload a car image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save uploaded file
    input_path = "input.jpg"
    with open(input_path, "wb") as f:
        f.write(uploaded_file.read())

    # Show the uploaded image
    st.image(input_path, caption="Uploaded Image", use_container_width=True)

    # Button to run detection
    if st.button("🔍 Detect"):
        # Clean old crops
        if os.path.exists("cropped_plates"):
            shutil.rmtree("cropped_plates")

        # Run YOLO inference
        results = model(input_path)
        results[0].save_crop("cropped_plates")

        # Show detection result (annotated image)
        st.image(input_path, caption="Detected License Plate", use_column_width=True)

        # OCR with EasyOCR
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
                    st.success(f"Car Plate: **{result[0][1]}**")
                else:
                    st.warning("Car Plate: Not recognized")
