import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

current_filter = None

def backgroundBlur(videoInput):
    # stub code
    return videoInput

def backgroundReplacement(videoInput):
    # stub code


    # Can use largely similar functions as above in initial stages of pipline, i.e. separating face out
    # then instead of blur we replace with our own background
    return cv2.GaussianBlur(videoInput, (15, 15), 1)

def faceDistortion(videoInput):
    # stub code


    return cv2.GaussianBlur(videoInput, (15, 15), 2)

def faceFilter(videoInput):
    # stub code


    return cv2.GaussianBlur(videoInput, (15, 15), 3)

def ourIdea(videoInput):
    # stub code


    return cv2.GaussianBlur(videoInput, (15, 15), 4)

def gstreamer_pipeline(
        capture_width=1920,
        capture_height=1080,
        display_width=960,
        display_height=540,
        framerate=30,
):
    return (
            "v4l2src ! "
            "video/x-raw, width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
            "videoflip method=horizontal-flip ! "  # Optional: Flip horizontally
            "videoconvert ! "
            "video/x-raw, width=(int)%d, height=(int)%d ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
            % (
                capture_width,
                capture_height,
                framerate,
                display_width,
                display_height,
            )
    )

def on_button_click(event, x, y, flags, param):
    global current_filter
    if event == cv2.EVENT_LBUTTONDOWN:
        if 10 <= x <= 110 and 10 <= y <= 60:
            current_filter = 'backgroundBlur'
        elif 120 <= x <= 220 and 10 <= y <= 60:
            current_filter = 'backgroundReplace'
        elif 230 <= x <= 330 and 10 <= y <= 60:
            current_filter = 'faceDistortion'
        elif 340 <= x <= 440 and 10 <= y <= 60:
            current_filter = 'faceFilter'
        elif 450 <= x <= 550 and 10 <= y <= 60:
            current_filter = 'ourIdea'

def create_buttons(frame):
    button_positions = [(10, 10, 100, 60), (120, 10, 220, 60), (230, 10, 330, 60), (340, 10, 440, 60), (450, 10, 550, 60)]
    button_names = ['Blur', 'Replace', 'Distort', 'Filter', 'Custom']

    for pos, name in zip(button_positions, button_names):
        # Create a rounded rectangle for the button with a slight shadow
        cv2.rectangle(frame, (pos[0], pos[1]), (pos[2], pos[3]), (220, 220, 220), -1)
        cv2.rectangle(frame, (pos[0] + 3, pos[1] + 3), (pos[2] + 3, pos[3] + 3), (200, 200, 200), -1)
        cv2.rectangle(frame, (pos[0], pos[1]), (pos[2], pos[3]), (0, 0, 0), 2, lineType=cv2.LINE_AA, shift=0)

        # Add text label
        cv2.putText(frame, name, (pos[0] + 15, pos[1] + 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    return frame

def show_camera():
    window_title = "Aidan and Sean Filter Output"

    # print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(0)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(window_title, on_button_click)
            while True:
                ret_val, frame = video_capture.read()

                ## our code starts here
                if current_filter == 'backgroundBlur':
                    frame = backgroundBlur(frame)
                elif current_filter == 'backgroundReplace':
                    frame = backgroundReplacement(frame)
                elif current_filter == 'faceDistortion':
                    frame = faceDistortion(frame)
                elif current_filter == 'faceFilter':
                    frame = faceFilter(frame)
                elif current_filter == 'ourIdea':
                    frame = ourIdea(frame)

                # Flip the frame horizontally
                frame = cv2.flip(frame, 1)

                create_buttons(frame)

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break


        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == '__main__':
    show_camera()

# could maybe make separate method to isolate person from input video, then feed that into functions for filtering, but this may be too much as well, can review later