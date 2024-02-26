import json
import onnxruntime as ort 
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
import torch
import cv2
import numpy as np
import pandas as pd
from torch import tensor
import sys
import os
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data.coco import mscoco_category2name
from src.data.staige_dataset import StaigeDataset
from torchvision.io import read_image
from torchvision import datapoints
from torchmetrics.detection.mean_ap import MeanAveragePrecision

HEIGHT = 2048
WIDTH = 6144
EPOCH = "00"
MODEL = f"/home/sergej/volume/RT-DETR/rtdetr_pytorch/soccer{EPOCH}.onnx"
ANNOTATIONS_FILE = "/home/sergej/volume/RT-DETR/rtdetr_pytorch/dataset/train3.csv"
CLASSES_FILE = "/home/sergej/volume/RT-DETR/rtdetr_pytorch/dataset/classes.csv"
output_directory = f'/home/sergej/volume/RT-DETR/rtdetr_pytorch/soccer_test/'
input_directory = "/home/sergej/volume/RT-DETR/rtdetr_pytorch/dataset"
THRESH = 0.4

class NumpyFloatValuesEncoder(json.JSONEncoder):
    def default(self, obj):
        # if isinstance(obj, np.float32):
        #     return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_coco_format(annotations, label_map):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    for label, id in label_map.items():
        categories_dict = {"id": id, "name": label}
        coco_format["categories"].append(categories_dict)
    
    img_paths = annotations.iloc[:, 0].unique()
    img_ids = dict(zip(img_paths, range(len(img_paths))))
    for idx, image in enumerate(img_paths):
        img_path = os.path.join(input_directory, image)
        image_dict = {"id": idx, "file_name": img_path, "height": HEIGHT, "width": WIDTH}
        coco_format["images"].append(image_dict)
    
    for idx, row in annotations.iterrows():
        img_path, x1, y1, x2, y2, class_name = row
        
        image_id = img_ids[img_path]
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        label_id = label_map[class_name]
        
        annotations_dict = {"id": idx, "image_id": image_id, "category_id": label_id, "bbox": bbox, "iscrowd": 0, "area": float((x2-x1)*(y2-y1))}
        coco_format["annotations"].append(annotations_dict)
    
    return coco_format

def main():
    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])#"CUDAExecutionProvider", 
    
    annotations = pd.read_csv(ANNOTATIONS_FILE, header=None)
    
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
    staige_labels2coco_id = {k: mscoco_name2category[v] for k, v in staige_labels2coco_name.items()}
    
    coco_format = get_coco_format(annotations, staige_labels2coco_id)
    
     # Save targets to JSON file
    output_file = "targets_coco_format.json"
    with open(output_file, "w") as f:
        json.dump(coco_format, f)
    
    size_tensor = torch.tensor([[WIDTH, HEIGHT]])
    preds = []
    id_counter = 0
    for i in range(3):
        target = coco_format["images"][i]
        img_path = target["file_name"]
        full_im = Image.open(img_path).convert('RGB')
        full_im = full_im.resize((WIDTH, HEIGHT))
        img_tensor = ToTensor()(full_im)[None]
        
        ypred = sess.run(
            # output_names=['labels', 'boxes', 'scores'],
            output_names=None,
            input_feed={'images': img_tensor.data.numpy(), "orig_target_sizes": size_tensor.data.numpy()}
        )
        
        labels, boxes, scores = ypred

        scr = scores[0]
        lab = labels[0][scr > THRESH]
        box = boxes[0][scr > THRESH]
        scr = scores[0][scr > THRESH]
        
        for index, b in enumerate(box):
            l = lab[index]
            s = scr[index]
            b = b.astype(float)
            area = (b[2] - b[0])*(b[3]-b[1])
            # pred_dict = {"id": id_counter, "image_id": int(target["id"]), "area": float(area), "category_id": l, "iscrowd": 0, "bbox": list(b)}
            pred_dict = {"image_id": int(target["id"]), "category_id": int(l), "bbox": list(b), "score": float(s)}
            id_counter += 1
            preds.append(pred_dict)
    
    # ds = StaigeDataset(ANNOTATIONS_FILE, input_directory, None)
    # img, tar = ds.__getitem__(0)
    # target_coco = metric._get_coco_format(labels=[tar["labels"]], boxes=[tar["boxes"]], masks=None, scores=None, crowds=[tar["iscrowd"]], area=[tar["area"]])
    # target_json = json.dumps(target_coco, indent=4)
    # with open(f"coco0_target.json", "w") as f:
    #     f.write(target_json)
    # metric.tm_to_coco("coco0")
    

    
   
        
    # Save preds to JSON file
    output_file = "preds_coco_format.json"
    with open(output_file, "w") as f:
        json.dump(preds, f, cls=NumpyFloatValuesEncoder)
    
    metric = MeanAveragePrecision()
    
    # Calc metric from coco Json
    p, t = MeanAveragePrecision.coco_to_tm(
        "preds_coco_format.json",
        "targets_coco_format.json",
        iou_type="bbox"
    )  
    
    metric.update(p, t)
    metric.compute()
    
    print("done")
    

if __name__ == '__main__':
    main()