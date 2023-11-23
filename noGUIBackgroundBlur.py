import cv2
import numpy as np

background = cv2.resize(cv2.imread('./backgrounds/1.jpg'), (960, 540))
face_cascade = cv2.cuda.CascadeClassifier_create('./haarcascades_cuda/haarcascade_frontalface_default.xml')
filter_image = cv2.imread('./images/1.jpeg')

gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, -1, (25,25), 5)

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

def backgroundBlur(frame):
    result = None
    image_gpu = cv2.cuda_GpuMat()   #declaring CUDA object into which we can pass images for processing with onboard GPU
    image_gpu.upload(frame)
    blurred_image_gpu = gaussian_filter.apply(image_gpu)
    blurred_frame = blurred_image_gpu.download()

    gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)

    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    faces_gpu = face_cascade.detectMultiScale(gray_gpu) #can't seem to alter parameters of this such as scale factor etc. 
    faces = faces_gpu.download()
    
    if faces is not None:
        for (x, y, w, h) in faces[0]:
            # Remove blur within the bounding box
            center_coordinates = x + w //2, y + h // 2
            radius = w // 2

            mask = np.zeros_like(frame)
            cv2.circle(mask, center_coordinates, radius, (255, 255, 255), -1)

            circular_region = cv2.bitwise_and(frame, mask)

            inverse_mask = cv2.bitwise_not(mask)

            destination_image = cv2.bitwise_and(blurred_frame, inverse_mask)

            result = cv2.add(destination_image, circular_region)

            blurred_frame = result

            # blurred_frame
            # cv2.circle(blurred_frame, center_coordinates, radius, (0, 255, 255), 2)

            # blurred_frame[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
            # cv2.rectangle(blurred_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return blurred_frame
    # return result

def show_camera():
    window_title = "Aidan and Sean"
    print(gstreamer_pipeline())

    videoCapture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if videoCapture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

            while True:
                ret, frame = videoCapture.read()  # Read a frame from the camera
                #logic here if doing multiple filters in single function
                processedFrame = backgroundBlur(frame)

                

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
