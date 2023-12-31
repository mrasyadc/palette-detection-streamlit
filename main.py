import cv2
import numpy as np
import tensorflow as tf
import time
from ultralytics import YOLO
import math

# https://dipankarmedh1.medium.com/real-time-object-detection-with-yolo-and-webcam-enhancing-your-computer-vision-skills-861b97c78993

# Initialize the video capture object (use 0 for the default camera)
cap = cv2.VideoCapture(0)

model_dir = "../ResNet50V2_Dense512_KFold10_20.model"
model = tf.keras.models.load_model(model_dir)
class_labels = ['60_Pola1_11', '60_Pola1_12', '60_Pola2_11', '60_Pola2_12', '63_Pola1_11', '63_Pola1_12', '63_Pola2_11', '63_Pola2_12']

# defined YOLOV8 model
model_yolo_dir = "../yolov8-full-best.pt"

# model
modelYolo = YOLO(model_yolo_dir)

# object classes
classNames = ["Varian 60 ml", "Varian 63 ml"]

# Variables for FPS calculation
start_time = time.time()
frame_count = 0

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Preprocess the frame for model prediction
    input_frame = cv2.resize(frame, (160, 320))
    

    image = np.array(input_frame) / 255.0 # Normalize the pixel values to the range of [0, 1]
    image = np.expand_dims(image, axis=0) 

    results = modelYolo(frame, stream=True)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # put box in cam
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])

            # object details
            org = [x1, y1]
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(frame, f'{classNames[cls]} {str(confidence*100)}%', org, font, fontScale, color, thickness) 

    # Perform model inference
    predictions = model.predict(image)

    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    confidence = predictions[0][predicted_class_index]

    # Calculate and display FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time
    cv2.putText(frame, f"FPS: {fps:.2f}", (frame.shape[1] - 150, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the classification result on the frame
    cv2.putText(frame, f"{predicted_class_label} with confidence {confidence*100}%", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Palette Pattern Detection and Classification', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
