import cv2
import numpy as np

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
    distorted_face = frame
    image_gpu.upload(frame)

    gray_gpu = cv2.cuda.cvtColor(image_gpu, cv2.COLOR_BGR2GRAY)

    #faces_gpu = face_cascade.detectMultiScale(gray_gpu)
    eye_gpu = eye_cascade.detectMultiScale(gray_gpu) 

    #faces = faces_gpu.download()
    eyes = eye_gpu.download()
    if eyes is not None:
        for (x, y, w, h) in eyes[0]:
            cv2.rectangle(distorted_face, (x,y), (x+w, y+h), (0,255,0), 2)

    # if faces is not None:
    #     for (x, y, w, h) in faces[0]:
    #         print("eh")
            # center_coordinates = x + w // 2, y + h // 2
            # radius = round(w / 1.4) 

            # #adding circular region to mask (for face)
            # mask = np.zeros_like(frame)         
            # cv2.circle(mask, center_coordinates, radius, (255, 255, 255), -1)

            # #adding rectangular region to mask (for shoulders & neck)
            # rect_x, rect_y, rect_w, rect_h = center_coordinates[0] - round(1.2*w), center_coordinates[1] - round(0.2*y)+ radius, round(2.4*w), 10*h #adjust as needed
            # cv2.rectangle(mask, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (255, 255, 255), thickness=-1)

            # # cv2.imshow("Person mask", mask)  #for adjustment in debugging

            # #isolating person from input frame
            ## region = cv2.cuda.bitwise_and(frame, mask)

            # inverse_mask = cv2.cuda.bitwise_not(mask)
            # backgroundCopy_with_hole = cv2.cuda.bitwise_and(backgroundCopy, inverse_mask)
            # faceDistort = cv2.add(backgroundCopy_with_hole, region)
            
            # # backgroundCopy[y:y+h, x:x+w] = frame[y:y+h, x:x+w]
            # # cv2.rectangle(backgroundCopy, (x, y), (x + w, y + h), (0, 255, 255), 2)

    return distorted_face

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
