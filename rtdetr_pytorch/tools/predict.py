import onnxruntime as ort 
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
import torch
import cv2
import numpy as np
import pandas as pd
import sys
import os
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data.coco import mscoco_category2name
from src.data.staige_dataset import StaigeDataset
from torchvision.io import read_image
from torchvision import datapoints

EPOCH = "00"
MODEL = f"/home/sergej/volume/RT-DETR/rtdetr_pytorch/soccer{EPOCH}.onnx"
ANNOTATIONS_FILE = "/home/sergej/volume/RT-DETR/rtdetr_pytorch/dataset/train3.csv"
CLASSES_FILE = "/home/sergej/volume/RT-DETR/rtdetr_pytorch/dataset/classes.csv"
output_directory = f'/home/sergej/volume/RT-DETR/rtdetr_pytorch/soccer_test/'
input_directory = "/home/sergej/volume/RT-DETR/rtdetr_pytorch/dataset"

def get_item(idx, img_paths, img_dir, annotations, classes):
    relative_img_path = img_paths[idx]
    img_path = os.path.join(img_dir, relative_img_path)
    image = read_image(img_path)

    image_annotations = annotations[annotations.iloc[:, 0] == relative_img_path]
    bboxes = []
    labels = []
    areas = []
    iscrowds = []
    for row in image_annotations.iterrows():
        _, x1, y1, x2, y2, class_name = row[1]
        bboxes.append([float(x1), float(y1), float(x2), float(y2)])
        labels.append(int(classes[class_name]))
        areas.append((x2-x1)*(y2-y1))
        iscrowds.append(0)  # TODO: Add occluded property?
    bboxes_tensor = datapoints.BoundingBox(
        bboxes,
        format=datapoints.BoundingBoxFormat.XYXY,
        spatial_size=image.shape[1:],  # h w
    )
    labels_tensor = torch.tensor(labels)
    areas_tensor = torch.tensor(areas)
    iscrowds_tensor = torch.tensor(iscrowds)
    image_id = torch.tensor([idx])
    orig_size = torch.tensor([image.shape[2], image.shape[1]])
    size = torch.tensor([image.shape[2], image.shape[1]])
    target = {
        "boxes": bboxes_tensor,
        "labels": labels_tensor,
        "image_id": image_id,
        "area": areas_tensor,
        "iscrowd": iscrowds_tensor,
        "orig_size": orig_size,
        "size": size,
    }
    return image, target


def main():
    print("main")

    sess = ort.InferenceSession(MODEL, providers=["CUDAExecutionProvider"])#"CPUExecutionProvider", 
    
    annotations = pd.read_csv(ANNOTATIONS_FILE, header=None)
    img_paths = annotations.iloc[:, 0].unique()
    
    classes = {}
    with open(CLASSES_FILE, mode='r') as f:
        reader = csv.reader(f)
        classes = {rows[0]:rows[1] for rows in reader}
    mscoco_name2category = {v: k for k, v in mscoco_category2name.items()}
    staige_labels2coco_name = {
        "ball": "sports ball",
        "goalkeeper": "person",
        "player": "person",
        "referee": "person",
        "horse": "horse",
        "mounted horse": "horse",
        "vehicle": "car",
        "person": "person"
    }
    staige_labels2coco_label = {k: mscoco_name2category[v] for k, v in staige_labels2coco_name.items()}
    
    ds = StaigeDataset(ANNOTATIONS_FILE, input_directory, None)
    
    image, target = get_item(0, img_paths, input_directory, annotations, staige_labels2coco_label)
    print("done")
    
    

if __name__ == '__main__':
    main()