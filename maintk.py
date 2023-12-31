import tkinter as tk
from tkinter import Canvas
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import numpy as np
import tensorflow as tf

class CameraApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
        # Set initial width and height based on the window size
        self.width = 1080
        self.height = 720

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.canvas = tk.Canvas(window, width=self.width, height=self.height)

        self.canvas.pack(padx=10, pady=10)

        self.btn_start_camera = tk.Button(window, text="Start Camera", width=15, command=self.start_camera)
        self.btn_start_camera.pack(padx=10, pady=10)

        self.btn_snapshot = tk.Button(window, text="Snapshot", width=10, command=self.snapshot)
        self.btn_snapshot.pack(padx=10, pady=10)

        self.classification_label = tk.Label(window, text="")
        self.classification_label.pack(pady=10)

        self.is_camera_running = False
        self.snapshot_folder = "snapshots"
        self.create_snapshot_folder()

        self.model_dir = "../ResNet50V2_Dense512_KFold10_20.model"
        self.model = tf.keras.models.load_model(self.model_dir)
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle window close event
        self.window.mainloop()

    def start_camera(self):
        self.is_camera_running = not self.is_camera_running
        if self.is_camera_running:
            self.btn_start_camera.config(text="Stop Camera")
        else:
            self.btn_start_camera.config(text="Start Camera")

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.snapshot_folder, f"snapshot_{timestamp}.png")
            try:
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print(f"Snapshot saved: {filename}")
                self.classify_frame(frame)
            except Exception as e:
                print(f"Error saving snapshot: {e}")

    def create_snapshot_folder(self):
        if not os.path.exists(self.snapshot_folder):
            os.makedirs(self.snapshot_folder)

    def classify_frame(self, frame):
        input_frame = cv2.resize(frame, (160, 320))
    

        image = np.array(input_frame) / 255.0 # Normalize the pixel values to the range of [0, 1]
        image = np.expand_dims(image, axis=0) 

        predictions = self.model.predict(image)

        predicted_class_index = np.argmax(predictions)
        # predicted_class_label = class_labels[predicted_class_index]
        confidence = predictions[0][predicted_class_index]
        label = f"{predicted_class_index} ({confidence})"

        text = f"Predicted: {label} confidence: {confidence*100}%"
        self.classification_label.config(text=text)
        self.draw_text_on_frame(frame, text)

    def draw_text_on_frame(self, frame, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)
        line_type = 1

        # Get text size
        text_size = cv2.getTextSize(text, font, font_scale, line_type)[0]

        # Calculate position to center the text at the top
        text_x = int((self.width - text_size[0]) / 2)
        text_y = text_size[1] + 10  # 10 pixels below the top

        cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, line_type)

    def update(self):
        if self.is_camera_running:
            ret, frame = self.vid.read()
            if ret:
                # Resize frame to fit the window
                frame = cv2.resize(frame, (self.width, self.height))
                self.photo = self.convert_to_photo_image(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.classify_frame(frame)
        self.window.after(10, self.update)


    def convert_to_photo_image(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(Image.fromarray(frame))
        return photo

    def on_close(self):
        self.__del__()
        self.window.destroy()

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Set the video source (use 0 for the default camera)
video_source = 0

# Create a window and pass it to the CameraApp class
root = tk.Tk()
app = CameraApp(root, "Tkinter App with OpenCV Camera", video_source)
