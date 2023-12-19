import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import tensorflow as tf
import numpy as np

# Load your trained model
model_dir = "../ResNet50V2_Dense512_KFold10_20.model"
model = tf.keras.models.load_model(model_dir)

# Function to preprocess the frame and make predictions
def process_frame(frame):
    print("Processing frame...")
    print("Frame shape:", frame.shape)
    # Resize and preprocess the frame as needed for your model
    processed_frame = cv2.resize(frame, (224, 224))
    processed_frame = np.expand_dims(processed_frame, axis=0)  # Add batch dimension if needed
    image = image.astype(np.float32) / 255.0
    
    # Add a batch dimension
    image = np.expand_dims(image, axis=0)

    # Make predictions using the model
    predictions = model.predict(processed_frame)

    # Assuming the model outputs class probabilities
    predicted_class = np.argmax(predictions)
    
    return predicted_class

def main():
    st.title("Real-time Video Classification with Streamlit")

    # Create a unique key for the webrtc_streamer
    webrtc_key = "example_key"
    
    # Display the WebRTC streamer
    webrtc_ctx = webrtc_streamer(key=webrtc_key, media_stream_constraints={"video": True, "audio": False})

    # st.write(model)
    # st.write(webrtc_ctx.video_receiver)


if __name__ == "__main__":
    main()
