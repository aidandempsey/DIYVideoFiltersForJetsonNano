import cv2
import tkinter as tk
from tkinter import Button, Scale, HORIZONTAL
from PIL import Image, ImageTk
import numpy as np

def backgroundBlur(videoInput, blur_level):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(videoInput, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    mask = np.ones_like(videoInput) * 255

    for (x, y, w, h) in faces:
        cv2.rectangle(mask, (x, y), (x + w, y + h), (0, 0, 0), -1)

    blurred_background = cv2.GaussianBlur(videoInput, (15, 15), blur_level)
    return cv2.bitwise_and(blurred_background, mask) + cv2.bitwise_and(videoInput, 255 - mask)

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
        blur_slider.pack()
        blur_slider.set(3)
    else:
        blur_slider.pack_forget()

    update_button_appearance()

def update_button_appearance():
    for button_name, filter_func in filters:
        button = button_dict[button_name]

        if button_name == current_filter:
            button.config(bg="#19c37d", fg="#555555")
        else:
            button.config(bg="#555555", fg="#19c37d")

def createGuiAndCaptureVideo():
    global blur_slider
    blur_slider = None

    videoCapture = cv2.VideoCapture(0)

    original_width = 640
    original_height = 480
    new_width = int(original_width * 1.5)
    new_height = int(original_height * 1.5)

    root = tk.Tk()
    root.title("Aidan and Sean Filter Output")
    root.configure(bg="#333333")

    canvas = tk.Canvas(root, width=new_width, height=new_height)
    canvas.pack()
    canvas.configure(bg="#333333")

    buttonFrame = tk.Frame(root)
    buttonFrame.pack(side=tk.BOTTOM, pady=0)

    global button_dict
    button_dict = {}

    filters = [
        ("Normal", removeFilters),
        ("Blur", backgroundBlur),
        ("Replace", backgroundReplacement),
        ("Distort", faceDistortion),
        ("Filter", faceFilter),
        ("Custom", ourIdea)
    ]

    for filter_name, filter_func in filters:
        button = Button(buttonFrame, text=filter_name, command=lambda name=filter_name: on_button_click(name), bg="#555555", fg="#19c37d", font=("Helvetica", 16))
        button.pack(side=tk.LEFT, expand=True)
        button_dict[filter_name] = button

    blur_slider = Scale(root, from_=1, to=5, orient=HORIZONTAL, length=new_width, sliderlength=20, showvalue=0, resolution=1)

    on_button_click("Normal")

    while True:
        ret, frame = videoCapture.read()
        if not ret:
            break

        if current_filter == "Blur":
            blur_level = blur_slider.get()
            frame = filters[next(i for i, (name, _) in enumerate(filters) if name == current_filter)][1](frame, blur_level)
        else:
            frame = filters[next(i for i, (name, _) in enumerate(filters) if name == current_filter)][1](frame)

        frame = cv2.resize(frame, (new_width, new_height))

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img = ImageTk.PhotoImage(image=frame_pil)

        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        root.update()

    videoCapture.release()

if __name__ == "__main__":
    current_filter = "Normal"
    filters = []
    button_dict = {}
    createGuiAndCaptureVideo()
