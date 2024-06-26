**FACE DETECTION WEB APP**
 The code creates a user-friendly web application for face detection using the MTCNN model. Users can upload images, detect faces within them, view the detected faces overlaid with bounding boxes, and download individual faces as image files. The workflow seamlessly integrates image processing functionalities with a simple and intuitive web interface, making it easy for users to interact with the face detection system.
![Screenshot 2024-04-21 143404](https://github.com/depak-kumar/face_detection_webaap/assets/129481998/d2f3f3f0-e890-426c-ab9f-7b941db5742b)

**Web Application Interface:** The code utilizes Streamlit, a library for creating web applications with Python. Streamlit simplifies the process of building interactive web interfaces by allowing developers to write Python scripts that are automatically converted into web applications.

**Upload Image**: Users are presented with an option to upload an image file (supported formats: jpg, png, jpeg). This functionality is facilitated by the st.file_uploader method provided by Streamlit. Users can select an image file from their local system.


**Image Display:** Once an image is uploaded, it is displayed in the web interface using the st.image method. The original image is shown with its caption.


**Face Detection:** Upon uploading an image, users have the option to click on the "Detect Faces" button. When clicked, the detect_faces_mtcnn function is called, which uses the MTCNN (Multi-Task Cascaded Convolutional Neural Network) model to detect faces within the uploaded image. Detected faces are outlined with bounding boxes.

**Display Detected Faces:** The image with bounding boxes around the detected faces is displayed in the web interface using st.image. Additionally, the number of detected faces is shown using st.markdown.

**Download Faces**: For each detected face, a download button is provided. Clicking on these buttons triggers the download_faces function, which saves each detected face as a separate image file. Users can download individual faces by clicking on the respective download buttons.

**Interaction:** The interaction with the web application is handled through Streamlit's built-in widgets such as file uploader, buttons, and image display functions. Users interact with the application by uploading images, clicking on buttons to trigger actions like face detection and face download, and viewing the results displayed in the interface.
