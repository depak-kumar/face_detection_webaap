import streamlit as st
import cv2
import numpy as np
from PIL import Image
from mtcnn import MTCNN

# Function to perform face detection using MTCNN
def detect_faces_mtcnn(image):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    for face in faces:
        x, y, w, h = face['box']
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return image, faces

# Function to download images with detected faces
def download_faces(image, faces):
    tempdir = "detected_faces"
    os.makedirs(tempdir, exist_ok=True)
    for i, face in enumerate(faces):
        x, y, w, h = face['box']
        face_img = image[y:y+h, x:x+w]
        face_pil = Image.fromarray(face_img)
        face_pil.save(os.path.join(tempdir, f'detected_face_{i+1}.jpg'))
        st.download_button(label=f"Download Face {i+1}", data=open(os.path.join(tempdir, f'detected_face_{i+1}.jpg'), 'rb'))

# Main function
def main():
    st.title("Face Detection with MTCNN")

    # Upload image option
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Original Image', use_column_width=True)
        image_cv = np.array(image)
        if st.button("Detect Faces"):
            result_image, faces = detect_faces_mtcnn(image_cv)
            st.image(result_image, caption='Image with Detected Faces', use_column_width=True)
            st.markdown(f"Number of detected faces: {len(faces)}")
            for i, face in enumerate(faces):
                if st.button(f"Download Face {i+1}"):
                    download_faces(image_cv, [face])

if __name__ == "__main__":
    main()
