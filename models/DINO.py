from groundingdino.util.inference import load_model, predict, annotate
import cv2
import groundingdino.datasets.transforms as T
from PIL import Image
import numpy as np
import torch
from torchvision.ops import box_convert


class DINO():
    def __init__(self, threshold = 0.35):
        self.model = load_model("models/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                                "models/GroundingDINO/weights/groundingdino_swint_ogc.pth", device='cpu')
        self.threshold = threshold

    def load_image(self, image_path):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if isinstance(image_path, str):
            image_source = Image.open(image_path).convert("RGB")
        else:
            image_source = image_path

        #image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)
        return image_source, image_transformed

    def exist(self, image, query):
        image_source, image = self.load_image(image)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=query,
            box_threshold=self.threshold,
            text_threshold=self.threshold
        )

        if len(phrases) == 0:
            return str(False)

        else:
            return str(True)

    def locate(self, image, query, count=False):
        image_source, image = self.load_image(image)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=query,
            box_threshold=self.threshold,
            text_threshold=self.threshold
        )

        if count:
            return str(len(phrases))

        else:
            if len(phrases) == 0:
                return image_source, None

            if len(phrases) > 1:
                idx = torch.argmax(logits)
                boxes = boxes[idx]

            w, h = image_source.size
            boxes = boxes * torch.Tensor([w, h, w, h])
            xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            xyxy = np.squeeze(xyxy)
            top_left_x, top_left_y, bottom_right_x, bottom_right_y = xyxy[0], xyxy[1], xyxy[2], xyxy[3]
            cropped_img = image_source.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
            # annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)

            return cropped_img, (top_left_x, top_left_y, bottom_right_x, bottom_right_y)

    def crop(self, image_source, direction, target=None):
        if isinstance(image_source, str):
            image_source = Image.open(image_source).convert("RGB")

        width, height = image_source.size
        if target:
            _, coordinates = self.locate(image_source, target)
        else:
            coordinates = None

        if 'left' in direction:
            if coordinates:
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = coordinates
                new_box = (0, 0, int((top_left_x + bottom_right_x) / 2), height - 1)
            else:
                new_box = (0, 0, int(width*0.6), height - 1)

        elif 'right' in direction:
            if coordinates:
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = coordinates
                new_box = (int((top_left_x + bottom_right_x) / 2), 0, width - 1, height - 1)
            else:
                new_box = (int(width*0.4), 0, width - 1, height - 1)

        elif 'above' in direction or 'top' in direction:
            if coordinates:
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = coordinates
                new_box = (0, 0, width - 1, int((top_left_y + bottom_right_y) / 2))
            else:
                new_box = (0, 0, width - 1, int(height*0.6))

        elif 'below' in direction or 'under' in direction or 'bottom' in direction:
            if coordinates:
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = coordinates
                new_box = (0, int((top_left_y + bottom_right_y) / 2), width - 1, height - 1)
            else:
                new_box = (0, int(height*0.4), width - 1, height - 1)

        else:
            return image_source

        cropped_img = image_source.crop(new_box)
        return cropped_img
