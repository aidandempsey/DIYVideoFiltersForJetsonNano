import cv2
import numpy as np

face_cascade = cv2.cuda.CascadeClassifier_create('./haarcascades_cuda/haarcascade_frontalface_default.xml')
filter_image = cv2.imread('./images/filter.jpeg')

image_gpu = cv2.cuda_GpuMat()   #declaring CUDA object into which we can pass images for processing with onboard GPU

def gstreamer_pipeline(
    capture_width=1280, #lowered from 1920x1080 for improved speed
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=60,   #increased from 30 again to improve speed
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

def faceFilter(frame):
    temp_filter_image = filter_image.copy()

    image_gpu.upload(frame)                             #move frame to onboard GPU for processing
    gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)

    faces_gpu = face_cascade.detectMultiScale(gray_gpu) 
    faces = faces_gpu.download()

    if faces is not None:
        for (x, y, w, h) in faces[0]:
            temp_filter_image = cv2.resize(temp_filter_image, (w,h))
            frame[y:y+h, x:x+w] = temp_filter_image
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return frame

def show_camera():
    window_title = "Aidan and Sean"
    print(gstreamer_pipeline())

    videoCapture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if videoCapture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

            while True:
                ret, frame = videoCapture.read()  # Read a frame from the camera
               
                processedFrame = faceFilter(frame)

                

                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, processedFrame)
                else:
                    break
                keycode = cv2.waitKey(10) & 0xFF

                if keycode == 27 or keycode == ord('q'): #allow user to quit gracefully
                    break
        finally:  
            videoCapture.release()
            cv2.destroyAllWindows()
    else:
        print("Error")

if __name__ == '__main__':
    show_camera()
