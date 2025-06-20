import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from .cal_mAP import json_map, json_map1
from .cal_PR import json_metric, json_metric1, metric, json_metric_top3


voc_classes = ("aeroplane", "bicycle", "bird", "boat", "bottle",
           "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor")
coco_classes = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
wider_classes = (
                "Male","longHair","sunglass","Hat","Tshiirt","longSleeve","formal",
                "shorts","jeans","longPants","skirt","faceMask", "logo","stripe")

nw_classes = ('clouds', 'sky', 'person', 'street', 'window', 'tattoo', 'wedding', 'animal', 'cat', 'buildings',
                     'tree', 'airport', 'plane', 'water', 'grass', 'cars', 'road', 'snow', 'sunset', 'railroad',
                     'train', 'flowers', 'plants', 'house', 'military', 'horses', 'nighttime', 'lake', 'rocks',
                     'waterfall', 'sun', 'vehicle', 'sports', 'reflection', 'temple', 'statue', 'ocean', 'town',
                     'beach', 'tower', 'toy', 'book', 'bridge', 'fire', 'mountain', 'rainbow', 'garden', 'police',
                     'coral', 'fox', 'sign', 'dog', 'cityscape', 'sand', 'dancing', 'leaf', 'tiger', 'moon', 'birds',
                     'food', 'cow', 'valley', 'fish', 'harbor', 'bear', 'castle', 'boats', 'running', 'glacier',
                     'swimmers', 'elk', 'frost', 'protest', 'soccer', 'flags', 'zebra', 'surf', 'whales', 'computer',
                     'earthquake', 'map')

class_dict = {
    "voc07": voc_classes,
    "coco": coco_classes,
    "wider": wider_classes,
    "nuswide": nw_classes,
}

def evaluation(result, types, ann_path):
    print("Evaluation")
    classes = class_dict[types]
    aps = np.zeros(len(classes), dtype=np.float64)

    if types=='nuswide':
        ann_json = ann_path
        pred_json = result
        for i, _ in enumerate(tqdm(classes)):
            ap = json_map1(i, pred_json, ann_json, types)
            aps[i] = ap
        OP, OR, OF1, CP, CR, CF1 = json_metric1(pred_json, ann_json, len(classes), types)
    else:
        ann_json = json.load(open(ann_path, "r"))
        pred_json = result
        for i, _ in enumerate(tqdm(classes)):
            ap = json_map(i, pred_json, ann_json, types)
            aps[i] = ap
        OP, OR, OF1, CP, CR, CF1 = json_metric(pred_json, ann_json, len(classes), types)

    print("mAP: {:4f}".format(np.mean(aps)))
    print("CP: {:4f}, CR: {:4f}, CF1 :{:4F}".format(CP, CR, CF1))
    print("OP: {:4f}, OR: {:4f}, OF1 {:4F}".format(OP, OR, OF1))
    return np.mean(aps),OP, OR, OF1, CP, CR, CF1



