(torch-1-8) PJLAB\xxx@pjnl104220145l:~/HDD/projects/isp/codes/DynamicISP/yolov3$ python val.py --data data/coco-2017.yaml --img 640 --conf 0.001 --iou 0.65 --batch-size 8 --save-json
val: data=data/coco-2017.yaml, weights=yolov3-spp.pt, batch_size=8, imgsz=640, conf_thres=0.001, iou_thres=0.65, max_det=300, task=val, device=, workers=8, single_cls=False, augment=False, verbose=False, save_txt=False, save_hybrid=False, save_conf=False, save_json=True, project=runs/val, name=exp, exist_ok=False, half=False, dnn=False
fatal: detected dubious ownership in repository at '/home/PJLAB/xxx/HDD/projects/isp/codes/DynamicISP/yolov3'
To add an exception for this directory, call:

        git config --global --add safe.directory /home/PJLAB/xxx/HDD/projects/isp/codes/DynamicISP/yolov3
YOLOv3 🚀 2023-8-15 Python-3.7.0 torch-1.8.1+cu102 CUDA:0 (NVIDIA GeForce GTX 1660 Ti, 5914MiB)

Fusing layers...
yolov3-spp summary: 269 layers, 62971933 parameters, 0 gradients
val: Scanning COCO/coco2017/val2017.cache... 4952 images, 48 backgrounds, 0 corrupt: 100%|██████████| 5000/5000 [00:00<?, ?it/s]
                 Class     Images  Instances          P          R      mAP50   mAP50-95: 100%|██████████| 625/625 [02:51<00:00,  3.65it/s]
                   all       5000      36335      0.723       0.61      0.664      0.474
Speed: 0.1ms pre-process, 29.3ms inference, 1.3ms NMS per image at shape (8, 3, 640, 640)
{'path': 'COCO/coco2017', 'train': 'COCO/coco2017/train2017.txt', 'val': 'COCO/coco2017/val2017.txt', 'test': 'COCO/coco2017/test-dev2017.txt', 'names': {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}, 'download': "from utils.general import download, Path\n\n\n# Download labels\nsegments = False  # segment or box labels\ndir = Path(yaml['path'])  # dataset root dir\nurl = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/'\nurls = [url + ('coco2017labels-segments.zip' if segments else 'coco2017labels.zip')]  # labels\ndownload(urls, dir=dir.parent)\n\n# Download data\nurls = ['http://images.cocodataset.org/zips/train2017.zip',  # 19G, 118k images\n        'http://images.cocodataset.org/zips/val2017.zip',  # 1G, 5k images\n        'http://images.cocodataset.org/zips/test2017.zip']  # 7G, 41k images (optional)\ndownload(urls, dir=dir / 'images', threads=3)\n", 'nc': 80}

Evaluating pycocotools mAP... saving runs/val/exp2/yolov3-spp_predictions.json...
loading annotations into memory...
Done (t=0.19s)
creating index...
index created!
Loading and preparing results...
DONE (t=2.03s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=38.51s).
Accumulating evaluation results...
DONE (t=6.61s).
# 640
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.477
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.669
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.520
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.307
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.525
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.608
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.365
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.602
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.657
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.500
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.706
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.795
Results saved to runs/val/exp2


# 512
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.655
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.505
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.281
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.514
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.630
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.360
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.584
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.636
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.445
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.692
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.802
Results saved to runs/val/exp3



python train.py --weights yolov3-spp.pt --cfg models/yolov3-spp.yaml --data data/coco-2017.yaml --batch-size 4 --imgsz 512