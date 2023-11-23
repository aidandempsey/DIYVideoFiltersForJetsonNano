import cv2

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

def face_and_upperbody_detect():
    window_title = "Sean + Aidan"
    upper_body_cascade = cv2.CascadeClassifier("./haarcascades_cuda/haarcascade_mcs_upperbody.xml")

    video_capture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

            while True:
                ret, frame = video_capture.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Create a blurred copy of the frame
                blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)

                upper_bodies = upper_body_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                # Detect upper bodies
                for (x, y, w, h) in upper_bodies:
                    # Remove blur within the bounding box
                    blurred_frame[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
                    cv2.rectangle(blurred_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                # Check to see if the user closed the window
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, blurred_frame)
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
        print("Unable to open camera")

if __name__ == "__main__":
    face_and_upperbody_detect()

