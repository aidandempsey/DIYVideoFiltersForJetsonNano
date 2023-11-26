# DIY Video Filters for Jetson Nano

## Project Overview

This repository contains Python scripts for real-time video processing filters on the Nvidia Jetson Nano. The filters include facial detection, object detection, background blurring, background replacement, face filtering, and facial distortion. The project leverages the Jetson Nano's GPU capabilities and employs CUDA-enabled functions from the OpenCV library for optimized image processing.

## Files

- **allFilters_GUI.py**: GUI script for applying filters in real-time.
- **noGUIBackgroundBlur.py**: Script for background blurring without GUI.
- **noGUIBackgroundReplace.py**: Script for background replacement without GUI.
- **noGUIFaceFilter.py**: Script for face filtering without GUI.
- **noGUIFaceDistort.py**: Script for facial distortion without GUI.
- **objectDetection.py**: Script for object detection filter without GUI.

## System Requirements

- Nvidia Jetson Nano
- CSI camera input (Raspberry Pi Camera Module 2)
- GStreamer pipeline for video feed
- OpenCV library with CUDA support

## Images and Models

- Images used for testing and demonstration are located in the `images` directory.
- Object detection model files (`deploy.prototxt` and `mobilenet_iter_73000.caffemodel`) are stored in the `models` directory.