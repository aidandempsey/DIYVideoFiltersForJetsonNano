import cv2
import numpy as np
import math


face_cascade = cv2.cuda.CascadeClassifier_create('./haarcascades_cuda/haarcascade_frontalface_default.xml')
eye_cascade = cv2.cuda.CascadeClassifier_create('./haarcascades_cuda/haarcascade_eye.xml')

gaussian_filter = cv2.cuda.createGaussianFilter(cv2.CV_8UC3, -1, (25,25), 5)
image_gpu = cv2.cuda_GpuMat()   #declaring CUDA object into which we can pass images for processing with onboard GPU

def gstreamer_pipeline(
    capture_width=1280, #lowered from 1920x1080 for improved speed
    capture_height=720,
    display_width=640,
    display_height=480,
    framerate=30,   #increased from 30 again to improve speed
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


def faceDistort(frame):
    image_gpu.upload(frame)
    gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)
    faces_gpu = face_cascade.detectMultiScale(gray_gpu)
    faces = faces_gpu.download()
   
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

def show_camera():
    window_title = "Aidan and Sean"
    print(gstreamer_pipeline())

    videoCapture = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)
    if videoCapture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

            while True:
                ret, frame = videoCapture.read()  # Read a frame from the camera
               
                processedFrame = faceDistort(frame)
                
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
