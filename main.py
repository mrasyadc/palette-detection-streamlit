# https://towardsdatascience.com/developing-web-based-real-time-video-audio-processing-apps-quickly-with-streamlit-7c7bcd0bc5a8

import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import numpy as np
import tensorflow as tf
import cv2
import queue
import random

model_dir = "../ResNet50V2_Dense512_KFold10_20.model"
model = tf.keras.models.load_model(model_dir)
class_labels = ['60_Pola1_11', '60_Pola1_12', '60_Pola2_11', '60_Pola2_12', '63_Pola1_11', '63_Pola1_12', '63_Pola2_11', '63_Pola2_12']  # Replace with your actual class labels

st.markdown("# Palette Pattern Detection")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    # preprocessed_image = preprocess_image(img)
    # predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    img = cv2.putText(img, class_labels[random.randint(0,7)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # send data to outside of this callback
    
    
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(key="example", video_frame_callback=video_frame_callback, media_stream_constraints={"video": True, "audio": False})

def preprocess_image(image): 
     # Resize the image to (160, 320)
    image = cv2.resize(image, (320, 160))
    
    # Convert the image to float and normalize the pixel values to the range of [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)
    return image