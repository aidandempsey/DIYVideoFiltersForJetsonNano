import cv2
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=2,
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

class VideoProcessor:
    def __init__(self, master):
        self.master = master
        self.master.title("Sean + Aidan")

        # Open the video source
        self.video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
        if not self.video_capture.isOpened():
            print("Unable to open camera")
            self.master.destroy()
            return

        # Create Canvas to display video
        self.canvas = tk.Canvas(master, width=960, height=540)
        self.canvas.pack()

        # Buttons
        self.blur_button = tk.Button(master, text="Background blur", command=self.blur_background)
        self.blur_button.pack(side=tk.LEFT)

        self.replace_button = tk.Button(master, text="Background replacement", command=self.replace_background)
        self.replace_button.pack(side=tk.LEFT)

        self.distort_button = tk.Button(master, text="Face distortion", command=self.distort_face)
        self.distort_button.pack(side=tk.LEFT)

        self.filter_button = tk.Button(master, text="Face filter", command=self.filter_face)
        self.filter_button.pack(side=tk.LEFT)

        self.custom_button = tk.Button(master, text="Custom", command=self.custom)
        self.custom_button.pack(side=tk.LEFT)

        # Initialize variables
        self.is_paused = False
        self.update()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.video_capture.release()
        self.master.destroy()

    def blur_background(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Apply blur
            blurred_frame = cv2.GaussianBlur(frame, (15, 15), 0)
            self.display_frame(blurred_frame)

    def replace_background(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Apply replace
            replaced_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.display_frame(replaced_frame, cmap='gray')

    def distort_face(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Apply distortion
            rows, cols, _ = frame.shape
            distorted_frame = frame.copy()
            for i in range(rows):
                distorted_frame[i] = frame[(i + 50) % rows]
            self.display_frame(distorted_frame)

    def filter_face(self):
        ret, frame = self.video_capture.read()
        if ret:
            # Apply filter
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray_frame, 50, 150)
            self.display_frame(edges, cmap='gray')

    def custom(self):
        ret, frame = self.video_capture.read()
        if ret:


            custom_frame = frame
            self.display_frame(custom_frame)

    def display_frame(self, frame, cmap=None):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.master.photo = photo

    def update(self):
        if not self.is_paused:
            ret, frame = self.video_capture.read()
            if ret:
                self.display_frame(frame)
        self.master.after(10, self.update)

def main():
    root = tk.Tk()
    app = VideoProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()

