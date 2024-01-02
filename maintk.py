import tkinter as tk
from tkinter import Canvas, filedialog, Text
import cv2
from PIL import Image, ImageTk
import os
from datetime import datetime
import numpy as np
import tensorflow as tf
import time
import ring

ring_player = ring.RingPlayer()

class CameraApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        
        self.video_source = video_source
        self.vid = cv2.VideoCapture(self.video_source)
       
        # Set initial width and height based on the window size
        self.width = 720
        self.height = 560

        self.vid.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.vid.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.classification_label = tk.Label(window, text="")
        self.classification_label.pack(pady=10)

        self.canvas = tk.Canvas(window, width=self.width, height=self.height)
        self.canvas.pack()

        self.btn_start_camera = tk.Button(window, text="Start Camera", width=15, command=self.start_camera)
        self.btn_start_camera.pack(side=tk.LEFT, padx=5)

        self.btn_snapshot = tk.Button(window, text="Snapshot", width=10, command=self.snapshot)
        self.btn_snapshot.pack(side=tk.LEFT, padx=5)

        self.btn_choose_model = tk.Button(window, text="Choose Model and Classes", command=self.choose_model)
        self.btn_choose_model.pack(side=tk.LEFT, padx=5)

        self.is_camera_running = False
        self.model_selected = False
        self.model = None

        self.max_tier_var = tk.StringVar(self.window)
        self.max_tier_var.set("1")  # Set the default value to 1

        self.tier_display_text = tk.Text(window, height=10, width=40)
        self.tier_display_text.pack(pady=10)

        self.label_max_tier = tk.Label(self.window, text="Choose max_tier:")
        self.label_max_tier.pack(pady=5)

        self.create_max_tier_dropdown()

        self.class_names = []

       
        self.current_tier = 1

         # Initialize the dictionary with None values
        self.tier_data = {str(i + 1): None for i in range(int(self.max_tier_var.get()))}

        

        self.snapshot_folder = "snapshots"
        self.create_snapshot_folder()
        
        self.update()
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)  # Handle window close event
        self.window.mainloop()

    def create_max_tier_dropdown(self):
        # Create max_tier dropdown only when the camera is stopped
        if not self.is_camera_running:
            self.dropdown_max_tier = tk.OptionMenu(self.window, self.max_tier_var, "1", "2", "3", "4", "5", "6", "7", "8", "9", "10")
            self.dropdown_max_tier.pack(pady=5)
            # Bind the Configure event to update_max_tier_handler
            self.dropdown_max_tier.bind("<Configure>", self.update_max_tier_handler)

    def destroy_max_tier_dropdown(self):
        # Destroy max_tier dropdown when the camera starts
        if hasattr(self, 'dropdown_max_tier'):
            self.dropdown_max_tier.destroy()

    def update_max_tier_handler(self, event):
        # Reset current_tier to 1 when max_tier value changes
        self.current_tier = 1

        # Update the tier_data dictionary with None values for the new max_tier
        new_max_tier = int(self.max_tier_var.get())
        self.tier_data = {str(i + 1): None for i in range(new_max_tier)}
        print("new max tier: ", new_max_tier)
        print("new tier data: ", self.tier_data)
        

    def update_max_tier_handler(self, event):
        # Allow updating max_tier only when the camera is stopped
        if not self.is_camera_running:
            # Reset current_tier to 1 when max_tier value changes
            self.current_tier = 1

            # Update the tier_data dictionary with None values for the new max_tier
            new_max_tier = int(self.max_tier_var.get())
            self.tier_data = {str(i + 1): None for i in range(new_max_tier)}
        else:
            # Display a message or handle the case where the camera is running
            print("Stop the camera before updating max_tier.")


    def start_camera(self):
        if not self.model:
            self.classification_label.config(text="Please select a model first")
            return
        else:
            self.is_camera_running = not self.is_camera_running
            if self.is_camera_running:
                self.btn_start_camera.config(text="Stop Camera")
                # Destroy the dropdown when the camera starts
                self.destroy_max_tier_dropdown()
            else:
                self.btn_start_camera.config(text="Start Camera")
                # Create the dropdown when the camera stops
                self.create_max_tier_dropdown()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            predicted_class = self.classification_label.cget("text")
            filename = os.path.join(self.snapshot_folder, f"snapshot_{predicted_class}_{timestamp}.png")
            try:
                cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                print(f"Snapshot saved: {filename}")
                
            except Exception as e:
                print(f"Error saving snapshot: {e}")

    def create_snapshot_folder(self):
        if not os.path.exists(self.snapshot_folder):
            os.makedirs(self.snapshot_folder)

    def classify_frame(self, frame):
        if self.current_tier > int(self.max_tier_var.get()):
            
            self.classification_label.config(text="This is the last tier. continue to next palette in 5 seconds")
            ring_player.play_final()
            time.sleep(5)
            self.current_tier = 1

            #reset tier data
            self.tier_data = {str(i + 1): None for i in range(int(self.max_tier_var.get()))}
            
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_frame = cv2.resize(frame, (160, 320))
    
        image = np.array(input_frame) / 255.0  # Normalize the pixel values to the range of [0, 1]
        image = np.expand_dims(image, axis=0) 

        predictions = self.model.predict(image)

        predicted_class_index = np.argmax(predictions)
        confidence = predictions[0][predicted_class_index]
        predicted_class = self.class_names[predicted_class_index]
        label = f"{predicted_class}_Confidence:{confidence*100:.2f}%"
        
        self.classification_label.config(text=label)

        if self.current_tier in [1,2,4,6]:
            if predicted_class == "Pola1_Benar":
                
                self.snapshot()
                self.tier_data[str(self.current_tier)] = f"{predicted_class} in {timestamp}"
                self.current_tier += 1
                self.classification_label.config(text="Model loaded successfully for tier {}".format(self.current_tier))
                ring_player.play_next()
                time.sleep(5)
            
        else:
            if predicted_class == "Pola2_Benar":
                
                self.snapshot()
                self.tier_data[str(self.current_tier)] = f"{predicted_class} in {timestamp}"
                self.current_tier += 1
                self.classification_label.config(text="Model loaded successfully for tier {}".format(self.current_tier))
                ring_player.play_next()
                time.sleep(5)
            
            

    def print_all_tiers_to_text_widget(self):
        self.tier_display_text.delete(1.0, tk.END)  # Clear the existing text
        
        for tier, data in self.tier_data.items():
            self.tier_display_text.insert(tk.END, f"Tier {tier}: {data}\n")

    def update(self):
        if self.is_camera_running:
            ret, frame = self.vid.read()
            if ret:
                print(self.max_tier_var.get())
                print(self.tier_data)
                # Resize frame to fit the window
                frame = cv2.resize(frame, (self.width, self.height))
                self.photo = self.convert_to_photo_image(frame)
                self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
                self.classify_frame(frame)
                
                self.print_all_tiers_to_text_widget()
        self.window.after(10, self.update)

    def choose_model(self):
        # Select model file
        model_file_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("Model files", "*.model")])
        if model_file_path:
            print(f"Selected model file: {model_file_path}")
            
            # Load the selected model
            self.model = tf.keras.models.load_model(model_file_path)
            self.current_tier = 1
            self.classification_label.config(text="Model loaded successfully")

            # Enable camera-related buttons
            if not self.is_camera_running:
                self.btn_start_camera.config(state=tk.NORMAL)
                self.btn_snapshot.config(state=tk.NORMAL)

            # Select class names file
            class_names_file_path = filedialog.askopenfilename(title="Select Class Names File", filetypes=[("Text files", "*.txt")])
            if class_names_file_path:
                print(f"Selected class names file: {class_names_file_path}")
                
                # Load class names from the selected file
                with open(class_names_file_path, "r") as class_names_file:
                    self.class_names = [line.strip() for line in class_names_file.readlines()]

                # Update the UI or perform any action with the loaded class names
                print("Class Names:", self.class_names)

            

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
