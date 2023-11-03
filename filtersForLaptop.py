import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

def backgroundBlur(videoInput):
    # stub code
    print('running background blur filter')



    return

def backgroundReplacement(videoInput):
    # stub code


    # Can use largely similar functions as above in initial stages of pipline, i.e. separating face out
    # then instead of blur we replace with our own background
    return

def faceDistortion(videoInput):
    # stub code


    return

def faceFilter(videoInput):
    # stub code


    return

def ourIdea(videoInput):
    # stub code


    return

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



def show_camera(filterType):
    window_title = "Aidan and Sean Filter Output"

    # print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(0)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()

                ## our code starts here
                if filterType == 'backgroundBlur':
                    backgroundBlur(frame)
                elif filterType == 'backgroundReplace':
                    backgroundReplacement(frame)
                elif filterType == 'faceDistortion':
                    faceDistortion(frame)
                elif filterType == 'faceFilter':
                    faceFilter(frame)
                elif filterType == 'ourIdea':
                    ourIdea(frame)


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
    # print('Aidan and Sean are sound men')
    print('Options to copy: backgroundBlur backgroundReplace faceDistortion faceFilter ourIdea')
    filterType = input('Enter filter type: \n')

    show_camera(filterType)



# could maybe make separate method to isolate person from input video, then feed that into functions for filtering, but this may be too much as well, can review later