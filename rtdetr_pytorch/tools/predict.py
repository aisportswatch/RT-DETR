import json
import onnxruntime as ort 
from PIL import Image, ImageDraw
from torchvision.transforms import ToTensor
import torch
import numpy as np
import pandas as pd
from torch import tensor
import sys
import os
import csv
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.data.coco import mscoco_category2name
from src.data.staige_dataset import StaigeDataset
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from tqdm import tqdm
import argparse
from clearml import Dataset, InputModel
import requests
from torch.utils.tensorboard import SummaryWriter


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

def download_file(url, filename):
    response = requests.get(url)
    if response.status_code == 200:  # Check if the request was successful
        with open(filename, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {filename} successfully.")
    else:
        print("Failed to download the file.")

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

def get_coco_format(annotations, label_map, input_directory, width, height):
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
        image_dict = {"id": idx, "file_name": img_path, "height": height, "width": width}
        coco_format["images"].append(image_dict)
    
    for idx, row in annotations.iterrows():
        img_path, x1, y1, x2, y2, class_name = row
        
        image_id = img_ids[img_path]
        bbox = [float(x1), float(y1), float(x2), float(y2)]
        label_id = label_map[class_name]
        
        annotations_dict = {"id": idx, "image_id": image_id, "category_id": label_id, "bbox": bbox, "iscrowd": 0, "area": float((x2-x1)*(y2-y1))}
        coco_format["annotations"].append(annotations_dict)
    
    return coco_format

def main(args, writer):
    height = args.height
    width = args.width
    model = args.model
    input_directory = args.input_dir
    output_directory = args.output_dir
    annotations_file = os.path.join(input_directory, args.annotations)
    thresh = args.thresh

    sess = ort.InferenceSession(model, providers=["CUDAExecutionProvider"])#"CUDAExecutionProvider", CPUExecutionProvider
    
    annotations = pd.read_csv(annotations_file, header=None)
    

    
    coco_format = get_coco_format(annotations, staige_labels2coco_id, input_directory, width, height)
    
     # Save targets to JSON file
    output_file = "targets_coco_format.json"
    with open(output_file, "w") as f:
        json.dump(coco_format, f)

    # Calculate predictions
    size_tensor = torch.tensor([[width, height]])
    preds = []
    for i in tqdm(range(0, len(coco_format["images"])), desc="Progress"):
        target = coco_format["images"][i]
        img_path = target["file_name"]
        full_im = Image.open(img_path).convert('RGB')
        full_im = full_im.resize((width, height))
        img_tensor = ToTensor()(full_im)[None]
        
        ypred = sess.run(
            # output_names=['labels', 'boxes', 'scores'],
            output_names=None,
            input_feed={'images': img_tensor.data.numpy(), "orig_target_sizes": size_tensor.data.numpy()}
        )
        
        labels, boxes, scores = ypred

        scr = scores[0]
        lab = labels[0][scr > thresh]
        box = boxes[0][scr > thresh]
        scr = scores[0][scr > thresh]
        
        for index, b in enumerate(box):
            l = lab[index]
            s = scr[index]
            b = b.astype(float)
            pred_dict = {"image_id": int(target["id"]), "category_id": int(l), "bbox": list(b), "score": float(s)}
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
    results = metric.compute()
    print(results)

    # Write to TensorBoard
    for key, value in results.items():
        if value.numel() != 1:
            continue
        writer.add_scalar(f'Test/{key}', value.item(), 0)

    # fig_, ax_ = metric.plot()
    # fig_.show()
    
    print("done")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-id', '-m', type=str, required=True)
    parser.add_argument('--output-dir', '-o', type=str, default="./results")
    parser.add_argument('--annotations', '-a', type=str, default="eval.csv")

    parser.add_argument('--width', type=int, default=6144)
    parser.add_argument('--height', type=int, default=2048)
    parser.add_argument('--thresh', '-t', type=float, default=0.4)

    parser.add_argument('--dataset-name', '-d', type=str, default="soccer_6k_single")
    parser.add_argument('--dataset-project', '-p', type=str, default="Pytorch Test")

    args = parser.parse_args()

    dataset_path = Dataset.get(
        dataset_name=args.dataset_name,
        dataset_project=args.dataset_project,
        alias="Soccer 6k dataset"
    ).get_local_copy()
    args.input_dir = dataset_path

    input_model = InputModel(model_id=args.model_id)
    args.model = input_model.get_local_copy()

    writer = SummaryWriter('runs')

    main(args, writer)