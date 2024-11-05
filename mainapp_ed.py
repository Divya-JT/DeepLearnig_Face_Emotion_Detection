import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
import pandas as pd


import tensorflow as tf
from PIL import Image
import cv2
import mediapipe as mp

# Set page config
st.set_page_config(
    page_title="Emotion Detection",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load the model once
model = load_model('model_4.h5')
model.load_weights("model_4.weights.h5")

# Define the labels for emotions (customize based on your dataset)
emotion_labels1 = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']


# Function to preprocess the uploaded image
def preprocess_image(image):
    img = image.resize((48, 48))  # Resize to the model's input size
    img_array = np.array(image)  # Convert to array
    img_array = img_array / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Streamlit app UI
st.title("Emotion Detection from Image")

# File uploader for image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg"])

# To detect face emotions
def detect_emotions(image):
    img = np.asarray(image)
    prediction = None
    #image gray
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except Exception as err:
        #st.write("Exception : ", err)
        img_gray = img

    face = detect_face(img_gray)
    if face.multi_face_landmarks:
        roi_gray = img_gray#img_gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)   # Add batch dimension
            roi = np.expand_dims(roi, axis=-1)   # Add channel dimension for grayscale
            #st.write(roi)
            prediction = model.predict(roi)[0]
           
    
    return prediction

def detect_face(img_arr):
    detect_face = None
    # Initialize MediaPipe Face Mesh
    mp_face_mesh = mp.solutions.face_mesh
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
        # Convert the image to RGB as Mediapipe works with RGB images
        image_rgb = np.array(image.convert('RGB'))

        # Process image with MediaPipe Face Mesh
        detect_face = face_mesh.process(image_rgb)

        #st.write("results : ", detect_face)
        #st.write("multi_face_landmarks : ", detect_face.multi_face_landmarks)

    return detect_face


if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    
    c1, c2 = st.columns([1, 4])
    c1.image(image, caption="Uploaded Image", use_column_width=True)

    emotion = detect_emotions(image)
    if(emotion is not None):
        predicted_class = int(np.argmax(emotion))

        predicted_emotion = emotion_labels1[predicted_class]
        confidence = np.max(emotion) * 100
        # Display the prediction results
        c2.subheader(f"Predicted Emotion: {predicted_emotion}")
        c2.subheader(f"Confidence: {confidence:.2f}%")
        d = { 'Emotion Labels': emotion_labels1, 'Emotion Values': emotion}
        df = pd.DataFrame(d)
        c2.dataframe(df)
        
    else:
        c2.subheader("Unable to find emotion")

    
    
