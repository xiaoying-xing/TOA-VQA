import torch
import torchvision
import cv2

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class faster_rcnn():
    def __init__(self, device):
        self.device = device
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        #device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        self.model.eval()

    def detect(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        img_tensor = torchvision.transforms.functional.to_tensor(image)
        img_tensor = img_tensor.to(self.device)
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            output = self.model(img_tensor)

        #boxes = output[0]['boxes'].cpu().numpy()
        labels = output[0]['labels'].cpu().numpy()
        scores = output[0]['scores'].cpu().numpy()
        result = []
        for label, score in zip(labels, scores):
            if score > 0.5:
                result.append(COCO_CLASSES[label])
        result = list(set(result))
        return str(result)

'''
model = faster_rcnn('cuda')
img_ids = ['2379737', '2370134', '2402116', '1592615', '2362368', '2407398', '2334758', '2360636', '2400072', '2368451', '2387481', '2325166', '2412075', '2383470', '2413467', '2386334', '2360401', '2391413', '2361166', '2352110', '2371802', '2315710', '2345261', '2400829', '2316956', '2362410', '2405249', '2375806', '2400645', '2403557', '2320507', '2326282']
for id in img_ids:
    path = '/files0/home/xiaoying/chatVQA/datasets/GQA/images/{}.jpg'.format(id)
    result = model.detect(path)
    print(result)
'''
