# Godentist Smart Dashboard: Indonesian Identity Card Scanner

## Overview

We developed a machine learning model for our Godentist Smart Dashboard to enhance the validation and extraction of information from Indonesian Identity Cards (KTP). This system employs two models:

1. Object Detection Model: Identifies each label and its corresponding value on the KTP and provides bounding boxes for each label.
2. Optical Character Recognition (OCR) Model: Extracts text within the bounding boxes using a custom font trained with Pytesseract.

   ![image](https://github.com/pt-periksa-gigi-indonesia/godentist-ocr-smart-dashboard/assets/88135952/0a29739f-5f91-4911-b7a1-dbeb93fd7efd)


## Dataset
The dataset for training our machine learning model was sourced from Roboflow. It consisted of 2500 open-source images of Indonesian Identity Cards, each with annotated labels to facilitate training.

## Model Development
We utilized YOLOv8 as the foundation for our object detection model. YOLO (You Only Look Once) is a state-of-the-art, real-time object detection system that performs detection in a single neural network pass. YOLOv8 improves upon its predecessors with enhanced accuracy and speed, making it suitable for real-time applications.

## Training and Classes
After training the YOLOv8 model, we defined five classes to be used in the OCR model:
- NIK: National Identification Number
- Name: Full name of the cardholder
- Place & Birthdate: Place and date of birth
- Address: Residential address
- Gender: Gender of the cardholder

## OCR Model: Pytesseract with Custom Font
For the OCR model, we employed Pytesseract, an OCR tool for Python, to extract text from the bounding boxes generated by the object detection model. The OCR model was trained on a custom font to ensure accurate text extraction from the KTP images.
