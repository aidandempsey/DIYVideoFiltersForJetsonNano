import cv2
import tkinter as tk
from PIL import Image, ImageTk

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
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
        self.create_button("Normal", self.normal)
        self.create_button("Background blur", self.blur_background)
        self.create_button("Background replacement", self.replace_background)
        self.create_button("Face distortion", self.distort_face)
        self.create_button("Face filter", self.filter_face)
        self.create_button("Custom", self.custom)

        # Initialize variables
        self.is_paused = False
        self.update()
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_button(self, text, command):
        button = tk.Button(self.master, text=text, command=command)
        button.pack(side=tk.LEFT)

    def on_closing(self):
        self.video_capture.release()
        self.master.destroy()

    def process_frame(self, frame, effect_function):
        ret, processed_frame = effect_function(frame)
        if ret:
            self.display_frame(processed_frame)

    def normal(self, frame):
        return True, frame

    def blur_background(self, frame):
        # Implement background blur effect
        return True, frame

    def replace_background(self, frame):
        # Implement background replacement effect
        return True, frame

    def distort_face(self, frame):
        # Implement face distortion effect
        return True, frame

    def filter_face(self, frame):
        # Implement face filter effect
        return True, frame

    def custom(self, frame):
        # Implement custom effect
        return True, frame

    def display_frame(self, frame, cmap=None):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
        self.canvas.create_image(0, 0, image=photo, anchor=tk.NW)
        self.master.photo = photo

    def update(self):
        if not self.is_paused:
            ret, frame = self.video_capture.read()
            if ret:
                # Apply the selected effect to the frame
                self.process_frame(frame, self.active_effect)
        self.master.after(10, self.update)

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessor(root)
    root.mainloop()

