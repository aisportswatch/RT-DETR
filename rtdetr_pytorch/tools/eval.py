from torchmetrics.detection.mean_ap import MeanAveragePrecision

PREDS = "preds_coco_format.json"
TARGETS = "targets_coco_format.json"

def main():
    metric = MeanAveragePrecision()
    
    # Calc metric from coco Json
    p, t = MeanAveragePrecision.coco_to_tm(
        PREDS,
        TARGETS,
        iou_type="bbox"
    )  
    
    metric.update(p, t)
    results = metric.compute()
    
    print(results)
    
if __name__ == '__main__':
    main()