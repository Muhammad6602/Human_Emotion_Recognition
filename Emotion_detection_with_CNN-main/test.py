import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras.models import model_from_json
import threading
import os
from PIL import Image, ImageTk

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Load emotion detection model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# Function to detect emotions in video
def detect_emotions(video_path, start_button, save_button):
    cap = cv2.VideoCapture(video_path)
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Save photo if emotion changes
            if frame_counter % 10 == 0:
                file_name = f"Tracked_emotions/emotion_{frame_counter}.jpg"
                cv2.imwrite(file_name, frame)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()
    start_button.config(state=tk.NORMAL)  # Enable start button after video finishes
    save_button.config(state=tk.NORMAL)  # Enable save button after video finishes

# Function to browse for a video file
def browse_file(start_button, save_button):
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4")])
    if file_path:
        start_button.config(state=tk.NORMAL)  # Enable start button after browsing
        save_button.config(state=tk.DISABLED)  # Disable save button after browsing
        start_button.video_path = file_path  # Store the video path in a custom attribute of the button

# Function to start emotion detection from laptop camera
def start_camera_detection(start_button, save_button):
    start_button.config(state=tk.DISABLED)  # Disable start button during camera detection
    save_button.config(state=tk.DISABLED)  # Disable save button during camera detection
    t = threading.Thread(target=detect_emotions_from_camera, args=(start_button, save_button))
    t.start()

# Function to detect emotions from laptop camera
def detect_emotions_from_camera(start_button, save_button):
    cap = cv2.VideoCapture(0)  # Open the laptop camera
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
        num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in num_faces:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
            roi_gray_frame = gray_frame[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            emotion_prediction = emotion_model.predict(cropped_img)
            maxindex = int(np.argmax(emotion_prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Save photo if emotion changes
            if frame_counter % 10 == 0:
                file_name = f"Tracked_emotions/emotion_{frame_counter}.jpg"
                cv2.imwrite(file_name, frame)

        cv2.imshow('Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()
    start_button.config(state=tk.NORMAL)  # Enable start button after camera detection
    save_button.config(state=tk.NORMAL)  # Enable save button after camera detection

# Function to start emotion detection from video file
def start_detection(start_button, save_button):
    video_path = start_button.video_path  # Get the stored video path
    if video_path:
        start_button.config(state=tk.DISABLED)  # Disable start button during video playback
        save_button.config(state=tk.DISABLED)  # Disable save button during video playback
        t = threading.Thread(target=detect_emotions, args=(video_path, start_button, save_button))
        t.start()

# Function to save emotions detected photos
def save_photos():
    # Create directory if not exists
    if not os.path.exists("Tracked_emotions"):
        os.makedirs("Tracked_emotions")

# Create the main tkinter window
root = tk.Tk()
root.title("Emotion Detector")
root.geometry("800x600")
root.configure(bg='#E6E6FA')  # Set background color to lavender

# Center the window on the screen
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x = (screen_width - 800) // 2
y = (screen_height - 600) // 2
root.geometry(f"800x600+{x}+{y}")

# Create a frame for the header
header_frame = tk.Frame(root, bg='#87CEEB')  # Set frame background color to sky blue
header_frame.pack(fill=tk.X)

# Create a label for the big title
title_label = tk.Label(header_frame, text="Emotion Detector", font=("Helvetica", 36, "bold"), foreground="white", bg='#87CEEB')  # Set label text color to white and background to sky blue
title_label.pack(pady=20)

# Create a label to display information
info_label = tk.Label(root, text="Click the 'Browse Video File' button to select a video file for emotion detection.", font=("Helvetica", 14), bg='#E6E6FA', fg='#333333')  # Set label text color to black and background to lavender
info_label.pack(pady=20)

# Function to change button color on hover
def on_enter(e):
    browse_button['background'] = '#FFA07A'
    start_button['background'] = '#FFA07A'
    camera_button['background'] = '#FFA07A'
    save_button['background'] = '#FFA07A'

# Function to change button color on leave
def on_leave(e):
    browse_button['background'] = '#FFC0CB'
    start_button['background'] = '#FFC0CB'
    camera_button['background'] = '#FFC0CB'
    save_button['background'] = '#FFC0CB'

# Create a button to browse for a video file
browse_button = tk.Button(root, text="Browse Video File", command=lambda: browse_file(start_button, save_button), font=("Helvetica", 14), bg='#FFC0CB', fg='white', relief=tk.FLAT, compound=tk.LEFT)  # Set button colors and add an icon
browse_button.pack(pady=10)

# Open and convert the JPEG image to PNG format
browse_icon = Image.open("file-folder-icon-sign-symbol-logo-vector.jpg")
browse_icon = browse_icon.resize((30, 30), Image.ANTIALIAS)
browse_icon = ImageTk.PhotoImage(browse_icon)

# Set the icon on the button
browse_button.config(image=browse_icon)
browse_button.image = browse_icon

# Create a button to start the application with an icon
start_button_icon = Image.open("start-button-icon-eps-260nw-1499498846.webp")
start_button_icon = start_button_icon.resize((30, 30), Image.ANTIALIAS)
start_button_icon = ImageTk.PhotoImage(start_button_icon)

start_button = tk.Button(root, text="Start Application", state=tk.DISABLED, command=lambda: start_detection(start_button, save_button), font=("Helvetica", 14), bg='#FFC0CB', fg='white', relief=tk.FLAT, compound=tk.LEFT)  # Set button colors and add an icon
start_button.pack(pady=10)

# Set the icon on the button
start_button.config(image=start_button_icon)
start_button.image = start_button_icon

# Create a button to start emotion detection from laptop camera
camera_button = tk.Button(root, text="Use Laptop Camera", command=lambda: start_camera_detection(start_button, save_button), font=("Helvetica", 14), bg='#FFC0CB', fg='white', relief=tk.FLAT)
camera_button.pack(pady=10)

# Open and convert the JPEG image to PNG format for the camera button icon
camera_button_icon = Image.open("vectorchef150503559.jpg")  # Replace "your_icon_image.jpg" with the path to your desired icon image file
camera_button_icon = camera_button_icon.resize((30, 30), Image.ANTIALIAS)
camera_button_icon = ImageTk.PhotoImage(camera_button_icon)

# Set the icon on the button
camera_button.config(image=camera_button_icon)
camera_button.image = camera_button_icon

# Create a button to save emotions detected photos
save_button = tk.Button(root, text="Save Emotions Photos", command=save_photos, font=("Helvetica", 14), bg='#FFC0CB', fg='white', relief=tk.FLAT)
save_button.pack(pady=10)

# Open and convert the JPEG image to PNG format for the save button icon
save_button_icon = Image.open("2c310gh.jpg")  # Replace "your_icon_image.jpg" with the path to your desired icon image file
save_button_icon = save_button_icon.resize((30, 30), Image.ANTIALIAS)
save_button_icon = ImageTk.PhotoImage(save_button_icon)

# Set the icon on the button
save_button.config(image=save_button_icon)
save_button.image = save_button_icon

# Run the tkinter event loop
root.mainloop()
