import cv2
import matplotlib.pyplot as plt
import numpy as np
import time


def backgroundBlur(videoInput):
    #stub code
    print('running background blur filter')


    #First need to detect face from input videoStream / frame from video
    #What we are working with depends on our implementation
    #Do we want to feed in raw camera input to each function and process this, or use main loop to chop up video and
    #send individual frames to each function?



def backgroundReplacement(videoInput):
    #stub code
    True

    #Can use largely similar functions as above in initial stages of pipline, i.e. separating face out
    # then instead of blur we replace with our own background



def faceDistortion(videoInput):
    #stub code
    True

def faceFilter(videoInput):
    #stub code
    True

def ourIdea(videoInput):
    #stub code
    True

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,     #Think we can adjust these parameters and lower input video quality if methods are running very badly and fix can't be found
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )
def show_camera(filterType):
    window_title = "Aidan and Sean Filter Output"
    print(gstreamer_pipeline(flip_method=0))
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():
        try:
            window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()

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





                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
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

    filterType = input('Enter filter type: \n')

    show_camera(filterType)

