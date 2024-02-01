import torch
from ultralytics import YOLO
from torchvision.transforms import functional as F
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt

names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 
        5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 
        10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 
        14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 
        20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 
        25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
        30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
        35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
        39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
        45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
        57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 
        63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 
        69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 
        75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}
categories=[]
def OD(image_path):
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8x.pt')
    # Load an image
    image = Image.open(image_path)
    # Convert the PIL Image to a numpy array
    image_np = np.array(image)
    ori_img = np.array(image)
    # Filter out detections with a score less than 0.5
    results = model(image, device='mps')  # Using Yolo to predict object detection
    result = results[0]  # get the first result. boxes, scores, classes, labels
    bboxes = np.array(result.boxes.xyxy.cpu(), dtype='int')  # get the bounding boxes
    classes = np.array(result.boxes.cls.cpu(), dtype='int')  # get the classes
    # Initialize an empty dictionary to store the extracted regions
    extracted_regions_dict = {}
    # Iterate through the bounding boxes and classes
    for bbox, cls in zip(bboxes, classes):
        x1, y1, x2, y2 = bbox
        # Get the category name
        category_name = names[cls]
        categories.append(names[cls])
        # Extract the region of the image corresponding to the bounding box
        region = ori_img[y1:y2, x1:x2]
        # Convert the numpy array back to a PIL Image
        region_image = Image.fromarray(region)
        # Store the extracted region in the dictionary using the category name as the key
        extracted_regions_dict[category_name] = region_image
    # Optionally, return the dictionary of extracted regions
    return categories, extracted_regions_dict, image_np