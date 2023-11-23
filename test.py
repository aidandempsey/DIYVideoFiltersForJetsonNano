import cv2
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk
import numpy as np

current_filter = "Normal"  # Set the initial filter state to "Normal"
filters = []
upper_body_cascade = cv2.CascadeClassifier("./haarcascades_cuda/haarcascade_mcs_upperbody.xml")
background = cv2.resize(cv2.imread('./backgrounds/1.jpg'), (960, 540))
face_cascade = cv2.CascadeClassifier("./haarcascades_cuda/haarcascade_frontalface_default.xml")
filter_image = cv2.imread('./images/1.jpeg')

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=60,
    flip_method=6,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def backgroundBlur(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    for (x, y, w, h) in upper_bodies:
        # Remove blur within the bounding box
        blurred_frame[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        cv2.rectangle(blurred_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return blurred_frame

def backgroundReplacement(frame):
    #upper_body_cascade = cv2.CascadeClassifier("./haarcascades_cuda/haarcascade_mcs_upperbody.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in upper_bodies:
        # Remove blur within the bounding box
        background[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
        cv2.rectangle(background, (x, y), (x + w, y + h), (0, 255, 255), 2)
    return background

def faceDistortion(frame):
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        frame[y:y+h, x:x+w] = cv2.flip(frame[y:y+h, x:x+w], 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return frame


def faceFilter(frame):
    #face_cascade = cv2.CascadeClassifier("./haarcascades_cuda/haarcascade_frontalface_default.xml")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    filter_image = filter_image #trying to avoid imread every time
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        filter_image = cv2.resize(filter_image, (w,h))
        frame[y:y+h, x:x+w] = filter_image
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return frame

def ourIdea(frame):
    return frame

def removeFilters(videoInput):
    return videoInput

def on_button_click(filter_name):
    global current_filter
    current_filter = filter_name

    # Update the button appearance based on the current filter
    update_button_appearance()

def update_button_appearance():
    for button_name, filter_func in filters:
        button = button_dict[button_name]

        if button_name == current_filter:
            # Invert button colors
            button.config(bg="#19c37d", fg="#555555")
        else:
            button.config(bg="#555555", fg="#19c37d")

# Function to create the GUI and capture video
def createGuiAndCaptureVideo():
    videoCapture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    # Original canvas dimensions
    width = 960
    height = 540

    # Create a GUI window
    root = tk.Tk()
    root.title("Aidan and Sean Filter Output")
    root.configure(bg="#333333")  # Dark gray background

    # Create a canvas to display video frames
    canvas = tk.Canvas(root, width=width, height=height)
    canvas.pack()
    canvas.configure(bg="#333333")

    # Calculate the new button frame height (increased by 50%)
    new_button_frame_height = int(90)

    # Create a row of buttons with an increased font size
    buttonFrame = tk.Frame(root)
    buttonFrame.pack(side=tk.BOTTOM, pady=new_button_frame_height // 2) 

    global button_dict  # Make button_dict global
    button_dict = {}  # A dictionary to store button references

    # Filter names and corresponding functions
    global filters  # Make filters global
    filters = [
        ("Normal", removeFilters),
        ("Blur Background", backgroundBlur),
        ("Replace Background", backgroundReplacement),
        ("Distort Face", faceDistortion),
        ("Filter Face", faceFilter),
        ("Surprise", ourIdea)
    ]

    # Create buttons for each filter with a larger font size
    for filter_name, filter_func in filters:
        button = Button(buttonFrame, text=filter_name, command=lambda name=filter_name: on_button_click(name), bg="#555555", fg="#19c37d", font=("Helvetica", 16))
        button.pack(side=tk.LEFT)
        button_dict[filter_name] = button

    # Update the button appearance based on the current filter (initiate the "Normal" state)
    update_button_appearance()

    while True:
        ret, frame = videoCapture.read()  # Read a frame from the camera
        if not ret:
            break

        # Apply the selected filter to the frame
        if current_filter:
            frame = filters[next(i for i, (name, _) in enumerate(filters) if name == current_filter)][1](frame)

        # Resize the frame to fit the canvas size
        frame = cv2.resize(frame, (width, height))

        # Convert the OpenCV frame to a PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(image=frame_pil)

        # Display the frame in the tkinter window
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        root.update()

    videoCapture.release()

# Start capturing and displaying video with filter selection
createGuiAndCaptureVideo()
