import streamlit as st
import cv2
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Mask Entry System")

st.title("🚪 Smart Mask Entry System")

# Load model
model = tf.keras.models.load_model("mask_detector.keras")

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# 📷 Camera input (works on phone)
img_file = st.camera_input("Take a photo")

if img_file is not None:
    # Convert image
    bytes_data = img_file.getvalue()
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    no_mask = False

    for (x, y, w, h) in faces:
        face = img[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = face / 255.0
        face = np.reshape(face, (1, 224, 224, 3))

        pred = model.predict(face, verbose=0)[0][0]

        if pred < 0.5:
            label = "ENTRY ALLOWED"
            color = (0, 255, 0)
        else:
            label = "NO ENTRY"
            color = (0, 0, 255)
            no_mask = True

        cv2.putText(img, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)

    # Show result
    if len(faces) == 0:
        st.warning("👤 No face detected")
    elif no_mask:
        st.error("🚫 NO MASK - ENTRY DENIED")
    else:
        st.success("✅ MASK DETECTED - ENTRY ALLOWED")

    st.image(img, channels="BGR")
