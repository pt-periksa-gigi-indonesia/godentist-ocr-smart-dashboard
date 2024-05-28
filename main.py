import cv2
import numpy as np
import pytesseract
import requests
from io import BytesIO
from PIL import Image
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import os

app = FastAPI()

# Allow CORS for all origins (you can restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ImageRequest(BaseModel):
    image_url: str


def convert_to_binary(image: np.array) -> np.array:
    """ Convert image to grayscale. """
    preprocessed_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return preprocessed_image


def thresholding_otsu(image: np.array) -> np.array:
    """ Apply Otsu's thresholding to binarize the image. """
    _, preprocessed_image = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return preprocessed_image


def resize_image_to_fixed_size(image: np.array, target_shape=(640, 640)) -> np.array:
    """ Resize image to the target shape using interpolation.

    Args:
        image (np.array): Input image.
        target_shape (tuple): Desired output shape (height, width).

    Returns:
        np.array: Resized image.
    """
    resized_image = cv2.resize(
        image, target_shape, interpolation=cv2.INTER_AREA)
    return resized_image


def preprocess_image_with_resize(image: np.array) -> np.array:
    """ Convert to binary, apply Otsu's thresholding, and resize to (640, 640). """
    image = convert_to_binary(image)
    # image = thresholding_otsu(image)
    image = resize_image_to_fixed_size(image, target_shape=(640, 640))
    return image


def extract_text(image, coords):
    """ Extract text from specified coordinates in the image using Tesseract OCR. """
    x1, y1, x2, y2 = coords
    crop_img = image[y1:y2, x1:x2]
    # Specify the language as 'ind' for Indonesian
    text = pytesseract.image_to_string(crop_img, lang='ind', config='--psm 6')
    return text.strip()


def download_image(url: str) -> np.array:
    response = requests.get(url)
    if response.status_code == 200:
        image = Image.open(BytesIO(response.content))
        return np.array(image)
    else:
        raise HTTPException(status_code=400, detail="Failed to download image")


# Get the current working directory
current_directory = os.getcwd()

# Use the current directory as the model directory
model_path = os.path.join(current_directory, 'model.pt')


@app.on_event("startup")
def load_model():
    global model
    model = YOLO(model_path)
    print("Model loaded successfully!")


@app.get("/")
async def read_root():
    return {"message": "Welcome to the OCR API!"}


@app.post("/detect/")
async def detect(request: ImageRequest):
    print("Received request to detect text in image...")
    image_url = request.image_url

    # Download the image from the provided URL
    try:
        image = download_image(image_url)
    except HTTPException as e:
        return JSONResponse(content={"detail": str(e.detail)}, status_code=e.status_code)

    # Preprocess the image
    img = preprocess_image_with_resize(image)
    img_display = np.stack((img,) * 3, axis=-1)  # Use for displaying results

    # Predict using the model
    results = model.predict(img_display)

    # Class names of interest
    class_names = ['NAMA', 'NIK', 'Tempat Tanggal Lahir',
                   'ALAMAT', 'JENIS KELAMIN']

    # Dictionary to store the best detection for each class name
    best_detections = {class_name: {"conf": 0, "text": ""}
                       for class_name in class_names}

    # Iterate through each detection in the results
    for result in results:
        for box in result.boxes:
            class_id = result.names[box.cls[0].item()]
            if class_id in class_names:
                conf = box.conf[0].item()
                if conf > best_detections[class_id]["conf"]:
                    coords = [round(x) for x in box.xyxy[0].tolist()]
                    text = extract_text(img_display, coords)
                    best_detections[class_id] = {
                        'conf': conf,
                        'text': text
                    }

    # Format the output as required
    formatted_output = {class_name: best_detections[class_name]['text']
                        for class_name in class_names if best_detections[class_name]['conf'] > 0}

    return JSONResponse(content=formatted_output)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
