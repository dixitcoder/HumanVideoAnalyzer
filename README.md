# HumanVideoAnalyzer

This repository provides a solution for detecting humans in video files and images using advanced AI-based models. The project leverages YOLOv5 and other pre-trained models to identify and annotate human faces and bodies in videos and images.

## Project Overview

The **VideoHumanDetection** project allows users to perform human detection on videos and images. It utilizes the YOLOv5 model for real-time object detection and annotation, specifically focused on detecting human faces and bodies. The repository includes sample files and tools for testing, processing, and visualizing the detection results.

## Repository Structure

The repository contains the following files and directories:


## Features

- **Real-Time Human Detection**: Detect humans, faces, and bodies in video streams with minimal delay.
- **Face and Body Detection**: Recognize faces and full bodies in both images and videos.
- **Multi-Person Detection**: Supports detection and tracking of multiple people within the same video frame.
- **Data Format Support**: Easily manage input and output data formats with the `dataFormat.json` file.
- **Pre-Trained Model**: Uses YOLOv5 pre-trained weights (`yolov5s.pt`) for accurate detection.
- **Cross-Platform Support**: Works on both Windows and Unix-based systems.

## Technologies Used

- **Python**: The primary programming language for backend logic.
- **YOLOv5**: Pre-trained model for human detection in images and videos.
- **OpenCV**: For video and image processing.
- **PyInstaller**: For packaging the application into standalone executables.
- **Flask**: Optional for a web interface (if applicable).

## Setup Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/dixitcoder/VideoHumanDetection.git
