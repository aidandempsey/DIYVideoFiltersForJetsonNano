import cv2
import tkinter as tk
from tkinter import Button, Scale  # Import the Scale widget
from PIL import Image, ImageTk
import numpy as np

current_filter = "Normal"  # Set the initial filter state to "Normal"
filters = []

def backgroundBlur(videoInput, blur_level):
    # Load a pre-trained Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Convert the input frame to grayscale for face detection
    gray = cv2.cvtColor(videoInput, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a mask for the background
    mask = np.ones_like(videoInput) * 255

    for (x, y, w, h) in faces:
        # Create a rectangular mask for each detected face
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 0), -1)

    # Apply Gaussian blur to the background with the specified level
    blurred_background = cv2.GaussianBlur(videoInput, (15, 15), blur_level)

    # Combine the blurred background with the sharp face and body using the mask
    result = cv2.bitwise_and(blurred_background, mask) + cv2.bitwise_and(videoInput, 255 - mask)

    return result

def backgroundReplacement(videoInput):
    return videoInput

def faceDistortion(videoInput):
    return cv2.GaussianBlur(videoInput, (15, 15), 2)

def faceFilter(videoInput):
    return cv2.GaussianBlur(videoInput, (15, 15), 3)

def ourIdea(videoInput):
    return cv2.GaussianBlur(videoInput, (15, 15), 4)

def removeFilters(videoInput):
    return videoInput

def on_button_click(filter_name):
    global current_filter
    current_filter = filter_name

    if filter_name == "Blur":
        blur_slider.set(3)  # Set the initial blur level when "Blur" is selected

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
    global blur_slider  # Declare blur_slider as a global variable

    blur_slider = None  # Initialize blur_slider to None

    videoCapture = cv2.VideoCapture(0)  # 0 represents the default camera (you can change it as needed)

    # Original canvas dimensions
    original_width = 640
    original_height = 480

    # Calculate the new canvas dimensions (increased by 50%)
    new_width = int(original_width * 1.5)
    new_height = int(original_height * 1.5)

    # Create a GUI window
    root = tk.Tk()
    root.title("Aidan and Sean Filter Output")
    root.configure(bg="#333333")  # Dark gray background

    # Create a canvas to display video frames (50% bigger)
    canvas = tk.Canvas(root, width=new_width, height=new_height)
    canvas.pack()
    canvas.configure(bg="#333333")

    # Calculate the new button frame height (increased by 50%)
    new_button_frame_height = int(60 * 1.5)

    # Create a row of buttons with an increased font size
    buttonFrame = tk.Frame(root)
    buttonFrame.pack(side=tk.BOTTOM, pady=new_button_frame_height // 2)  # Reduce the pady value for a smaller button frame

    global button_dict  # Make button_dict global
    button_dict = {}  # A dictionary to store button references

    # Filter names and corresponding functions
    global filters  # Make filters global
    filters = [
        ("Normal", removeFilters),
        ("Blur", backgroundBlur),
        ("Replace", backgroundReplacement),
        ("Distort", faceDistortion),
        ("Filter", faceFilter),
        ("Custom", ourIdea)
    ]

    # Create buttons for each filter with a larger font size
    for filter_name, filter_func in filters:
        if filter_name == "Blur":
            button = Button(buttonFrame, text=filter_name, command=lambda name=filter_name: on_button_click(name), bg="#555555", fg="#19c37d", font=("Helvetica", 16))
        else:
            button = Button(buttonFrame, text=filter_name, command=lambda name=filter_name: on_button_click(name), bg="#555555", fg="#19c37d", font=("Helvetica", 16))
        button.pack(side=tk.LEFT)
        button_dict[filter_name] = button

    # Create the blur slider with 6 discrete increments
    blur_slider = Scale(buttonFrame, from_=1, to=5, orient=tk.HORIZONTAL, length=new_width, sliderlength=20, label="Blur Level", font=("Helvetica", 10), resolution=1)
    blur_slider.pack(side=tk.BOTTOM)

    # Set the default value to 3
    blur_slider.set(3)

    # Set the initial filter to "Normal"
    on_button_click("Normal")

    while True:
        ret, frame = videoCapture.read()  # Read a frame from the camera
        if not ret:
            break

        # Apply the selected filter to the frame
        if current_filter == "Blur":
            blur_level = blur_slider.get()  # Get the blur level from the slider
            frame = filters[next(i for i, (name, _) in enumerate(filters) if name == current_filter)][1](frame, blur_level)
        else:
            frame = filters[next(i for i, (name, _) in enumerate(filters) if name == current_filter)][1](frame)

        # Resize the frame to fit the canvas size (50% bigger)
        frame = cv2.resize(frame, (new_width, new_height))

        # Convert the OpenCV frame to a PIL image
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(image=frame_pil)

        # Display the frame in the tkinter window
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        root.update()

    videoCapture.release()

# Start capturing and displaying video with filter selection
createGuiAndCaptureVideo()
