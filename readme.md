# EE551 – DIY Filters on Jetson Nano Project
- Sean Breen - 19410202
- Aidan Dempsey – 19458984

## Introduction/Problem Statement:
The aim of this assignment is to develop a set of real-time video processing filters to be implemented in Python on an Nvidia Jetson Nano camera feed. We will use traditional image processing techniques such as masking and bitwise pixel operations, while also integrating neural network methodologies to enhance our filters.

## Algorithms/System Design:
### Camera Input
We use a GStreamer pipeline string to take video input from the camera module, tweaking some of the parameters to optimize for speed. The width and height of both the video capture from the camera and the displayed video output were lowered for improved speed (1280x720 and 640x480, respectively). At the same time, the framerate is set to 60fps, instead of the standard 30fps.

### Facial Detection

To perform facial recognition, we utilized [haarcascade_frontalface_default.xml](./haarcascades_cuda//haarcascade_frontalface_default.xml). This XML file represents a pre-trained Haar Cascade Classifier model designed for detecting faces, with this specific implementation of the Haar Cascade Classifier optimized for CUDA. For more detailed facial recognition, we made use of [haarcascade_eye.xml](./haarcascades_cuda//haarcascade_eye.xml), also optimized for CUDA.

### Object Detection
To perform object detection, we took a deep neural network approach. The file [deploy.prototxt](./models/deploy.prototxt) defines the neural network's architecture, whereas the [mobilenet_iter_73000.caffemodel](./models/mobilenet_iter_73000.caffemodel) file contains the pre-trained weights. This specific model is based on “MobileNet”, a lightweight convolutional neural network architecture. 

### CUDA and GPU Acceleration
Each filter is optimized to capitalize on the Jetson Nano’s GPU capabilities, specifically leveraging CUDA. Each video frame is initially uploaded to the GPU’s memory, facilitating parallelized processing. From there, it is processed using OpenCV’s dedicated CUDA-enabled functions, ensuring the computationally intensive tasks are efficiently distributed across the GPU's parallel processing units. The image is then downloaded to the CPU for any potential additional CPU-based refinements, before being displayed in real time.

### GUI
The graphical user interface (GUI) was developed with Tkinter, a library that facilitates the creation of windows, frames, buttons, and other elements. Our specific GUI contained a panel of buttons that could be clicked to change the video processing effects in real-time. Ultimately, the idea of the GUI was scrapped as it created huge bottlenecks in terms of performance.

### Filters
#### Background Blur
First, we first make a copy of the frame to serve as the background. The original frame is then converted into grayscale and a Haar cascade is applied to identify any faces. For each detected face, a circular mask is created to isolate the face from the rest of the frame. This mask is positioned on the centre of the face and its diameter is determined by the dimensions of the detected face’s bounding box. An additional rectangular mask is added to cover the upper body which is combined with the circle to create a unified mask. Using a bitwise AND operation, the person can then be isolated from the frame.

## Testing and Results:

## Discussion: (analysis of implementation performance)

## Suggested Improvements: 
Running it on something with actual processing power. 

### Bibliogrpahy
- Haarcascade
- Opencv
- Numpy
- MobileNet
