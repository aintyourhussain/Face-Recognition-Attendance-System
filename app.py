import streamlit as st
import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime
from PIL import Image

st.set_page_config(page_title="Face Recognition Attendance", layout="centered")

st.title("📷 Face Recognition Attendance System")
st.write("Capture a picture and the system will mark attendance automatically.")

# ---- Load known faces ----
@st.cache_resource
def load_known_faces():
    known_face_encodings = []
    known_face_names = []

    # Load images and encodings
    zunair_image = face_recognition.load_image_file("faces/student1.jpeg")
    zunair_encoding = face_recognition.face_encodings(zunair_image)[0]

    student2_image = face_recognition.load_image_file("faces/student2.jpeg")
    student2_encoding = face_recognition.face_encodings(student2_image)[0]

    student3_image = face_recognition.load_image_file("faces/student3.jpeg")
    student3_encoding = face_recognition.face_encodings(student3_image)[0]

    student4_image = face_recognition.load_image_file("faces/student4.jpeg")
    student4_encoding = face_recognition.face_encodings(student4_image)[0]

    known_face_encodings = [zunair_encoding, student2_encoding, student3_encoding, student4_encoding]
    known_face_names = ["student1", "student2", "student3", "student4"]

    return known_face_encodings, known_face_names

known_face_encodings, known_face_names = load_known_faces()

# ---- Setup CSV for attendance ----
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
csv_filename = f"{current_date}.csv"

# Create file with header only once
if not os.path.exists(csv_filename):
    with open(csv_filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# Use session_state to remember who is already marked present
if "recorded_names" not in st.session_state:
    st.session_state.recorded_names = set()

st.write(f"📅 Attendance file for today: **{csv_filename}**")

# ---- Camera Input ----
image_data = st.camera_input("Take a picture to mark attendance")

if image_data is not None:
    # Read image from camera widget
    image = Image.open(image_data)
    image_np = np.array(image)  # PIL to numpy (RGB)

    # Optional: resize for speed (similar to your original code)
    small_frame = cv2.resize(image_np, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame  # already in RGB from camera_input

    # Detect faces and encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    found_names = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            found_names.append(name)

            # Mark attendance only if not already recorded
            if name not in st.session_state.recorded_names:
                current_time = datetime.now().strftime("%I:%M:%S %p")
                with open(csv_filename, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([name, current_date, current_time])

                st.session_state.recorded_names.add(name)

    # ---- Show results ----
    if found_names:
        st.success("✅ Attendance marked for:")
        for n in set(found_names):
            st.write(f"- **{n}**")
    else:
        st.warning("No known faces detected in this picture.")

    # Show the captured image
    st.image(image_np, caption="Captured Image", use_column_width=True)

# Show list of already recorded names
if st.session_state.recorded_names:
    st.write("📋 Already marked present today:")
    for n in st.session_state.recorded_names:
        st.write(f"- {n}")
