import torch
from ultralytics import YOLO
import pandas as pd
import cv2 

class ObjectDetector:
    def __init__(self, model_path):
        self.model =  YOLO('yolov8n.pt')
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')

    def detect_objects(self, image):
        # Preprocess image for YOLOv8
        raw_image = cv2.imread(r"C:\Users\Ashwin\Downloads\spacex.jpg", cv2.IMREAD_COLOR)
        image = self.model(raw_image, )

        print(image)

class DepthEstimator:
    def __init__(self, model_path):
        self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=True)
        self.model.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.eval()

    def estimate_depth(self, image, bbox):
        # Crop image based on bounding box
        cropped_image = image[bbox[1]:bbox[3], bbox[0]:bbox[2], :]

        # Preprocess cropped image for MiDaS
        # (Refer to MiDaS documentation for specific preprocessing steps)

        # Predict depth map using MiDaS
        with torch.no_grad():
            depth_map = self.model(cropped_image.unsqueeze(0))

        # Extract average depth value from the depth map
        average_depth = depth_map.mean().item()

        return average_depth