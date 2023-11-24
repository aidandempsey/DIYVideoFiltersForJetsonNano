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
        eyes = sorted(eyes[0], key=lambda x: x[2] * x[3], reverse = True)
        for i in range(min(2, len(eyes))):
            (x, y, w, h) = eyes[i]

            # original_points = np.array([[x,y], [x+w, y], [x, y+h], [x+w, y+h]], dtype=np.float32)
            # warped_points = np.array([[x+4,y+7], [x+w+6, y+8], [x+3, y+h+9], [x+w+15, y+h-17]], dtype=np.float32)
            
            # M = cv2.getPerspectiveTransform(original_points, warped_points)

            # #if works, can it be done in GPU?
            # warped_img = cv2.warpPerspective(frame, M, (frame.shape[1], frame.shape[0]))
            
            # distorted_face = warped_img
            # #cv2.rectangle(distorted_face, (x,y), (x+w, y+h), (0,255,0), 2)

            roi = frame[y:y+h, x:x+w]

            # Define the stretching parameters
            stretch_factor_x = 1.5
            stretch_factor_y = 0.8

            # Apply the affine transformation to the ROI
            rows, cols, _ = roi.shape
            pts_original = np.float32([[0, 0], [cols, 0], [0, rows]])
            pts_stretched = np.float32([[0, 0], [int(cols * stretch_factor_x), 0], [0, int(rows * stretch_factor_y)]])
            M = cv2.getAffineTransform(pts_original, pts_stretched)
            stretched_roi = cv2.warpAffine(roi, M, (int(cols * stretch_factor_x), int(rows * stretch_factor_y)))

            # Replace the original ROI with the stretched ROI in the image
            frame[y:y+h, x:x+w] = stretched_roi
            distorted_face = frame
    
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
