import cv2
import numpy as np
import tkinter as tk
import math
from tkinter import Button
from PIL import Image, ImageTk

face_cascade = cv2.CascadeClassifier('./haarcascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('./haarcascades_cuda/haarcascade_eye.xml')

current_filter = "Normal"  # Set the initial filter state to "Normal"
filters = []

net = cv2.dnn.readNetFromCaffe('./models/deploy.prototxt', './models/mobilenet_iter_73000.caffemodel')

CLASSES = ('background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

background = cv2.resize(cv2.imread("./images/background.jpg"), (640, 480))
filter_image = cv2.imread('./images/filter.jpeg')

def background_blur(frame):
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    gray = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray) 
    
    if faces is not None:
        for (x, y, w, h) in faces:
            # Creating bounding box with which to remove correct region from blurred frame and add in un-blurred person to output
            center_coordinates = x + w // 2, y + h // 2
            radius = round(w / 1.4) 

            #adding circular region to mask (for face)
            mask = np.zeros_like(frame)         
            cv2.circle(mask, center_coordinates, radius, (255, 255, 255), -1)

            #adding rectangular region to mask (for shoulders & neck)
            rect_x, rect_y, rect_w, rect_h = center_coordinates[0] - round(1.2*w), center_coordinates[1] - round(0.2*y)+ radius, round(2.4*w), 10*h #adjust as needed
            cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 255, 255), thickness=-1)

            # cv2.imshow("Person mask", mask)  #for adjustment in debugging

            #isolating person from input frame
            region = cv2.bitwise_and(frame, mask)

            #creating hole in blurred frame into which we fit unblurred person
            inverse_mask = cv2.bitwise_not(mask)
            blurred_frame_with_hole = cv2.bitwise_and(blurred_frame, inverse_mask)
            blurred_frame = cv2.add(blurred_frame_with_hole, region)

    return blurred_frame

def background_replacement(frame):
    background_copy = background.copy()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray) 

    if faces is not None:
        for (x, y, w, h) in faces:

            center_coordinates = x + w // 2, y + h // 2
            radius = round(w / 1.4) 

            #adding circular region to mask (for face)
            mask = np.zeros_like(frame)         
            cv2.circle(mask, center_coordinates, radius, (255, 255, 255), -1)

            #adding rectangular region to mask (for shoulders & neck)
            rect_x, rect_y, rect_w, rect_h = center_coordinates[0] - round(1.2*w), center_coordinates[1] - round(0.2*y)+ radius, round(2.4*w), 10*h #adjust as needed
            cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 255, 255), thickness=-1)

            # cv2.imshow("Person mask", mask)  #for adjustment in debugging

            #isolating person from input frame
            region = cv2.bitwise_and(frame, mask)

            inverse_mask = cv2.bitwise_not(mask)
            background_copy_with_hole = cv2.bitwise_and(background_copy, inverse_mask)
            background_copy = cv2.add(background_copy_with_hole, region)
            
    return background_copy

def face_distortion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
   
    img = frame
    
    if faces is not None:
        
        res = np.zeros((img.shape[0],img.shape[1],3),np.uint8)

        radius =200 # radius of filter
        angel=-90 * math.pi/180 # angel

        centerx=(float((img.shape[1]-1))/2)
        centery=(float((img.shape[0]-1))/2)

        # Move the definition of y and x inside the loop
        y, x = np.ogrid[:img.shape[0], :img.shape[1]]
        px, py = np.meshgrid(np.arange(img.shape[1]) - centerx, np.arange(img.shape[0]) - centery)

        dis = np.sqrt(px**2 + py**2)
        mask = dis <= radius

        a = angel * (1 - dis[mask] / radius)

        px[mask] = px[mask] * np.cos(a) - py[mask] * np.sin(a)
        py[mask] = px[mask] * np.sin(a) + py[mask] * np.cos(a)

        newx = np.clip(centerx + px, 0, img.shape[1]-1).astype(int)
        newy = np.clip(centery + py, 0, img.shape[0]-1).astype(int)

        res[y,x] = img[newy,newx]
        frame = res
    return frame


def face_filter(frame):
    temp_filter_image = filter_image.copy()
    gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray) 

    if faces is not None:
        for (x, y, w, h) in faces:
            temp_filter_image = cv2.resize(temp_filter_image, (w,h))
            frame[y:y+h, x:x+w] = temp_filter_image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return frame

def object_detection(frame):
    # Get the dimensions of the frame
    height, width = frame.shape[:2]

    # Convert the frame to a blob for the model
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)

    # Set the input to the model
    net.setInput(blob)

    # Run forward pass to get detections
    detections = net.forward()

    # Process the detections and draw bounding boxes on the frame
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.2:  # Adjust confidence threshold as needed
            class_id = int(detections[0, 0, i, 1])
            class_label = f"Object: {CLASSES[class_id]}"
            confidence_score = f"Confidence: {confidence:.2f}"

            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (start_x, start_y, end_x, end_y) = box.astype("int")

            # Draw the bounding box
            cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

            # Display class label and confidence score
            y = start_y - 15 if start_y - 15 > 15 else start_y + 15
            cv2.putText(frame, class_label, (start_x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.putText(frame, confidence_score, (start_x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    return frame


def remove_filters(video_input):
    return video_input

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
def create_gui_and_capture_video():
    video_capture = cv2.VideoCapture(0)

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
    button_frame = tk.Frame(root)
    button_frame.pack(side=tk.BOTTOM, pady=new_button_frame_height // 2) 

    global button_dict  # Make button_dict global
    button_dict = {}  # A dictionary to store button references

    # Filter names and corresponding functions
    global filters  # Make filters global
    filters = [
        ("Normal", remove_filters),
        ("Blur Background", background_blur),
        ("Replace Background", background_replacement),
        ("Distort Face", face_distortion),
        ("Filter Face", face_filter),
        ("Surprise", object_detection)
    ]

    # Create buttons for each filter with a larger font size
    for filter_name, filter_func in filters:
        button = Button(button_frame, text=filter_name, command=lambda name=filter_name: on_button_click(name), bg="#555555", fg="#19c37d", font=("Helvetica", 16))
        button.pack(side=tk.LEFT)
        button_dict[filter_name] = button

    # Update the button appearance based on the current filter (initiate the "Normal" state)
    update_button_appearance()

    while True:
        ret, frame = video_capture.read()  # Read a frame from the camera
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

    video_capture.release()

# Start capturing and displaying video with filter selection
create_gui_and_capture_video()
