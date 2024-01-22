from transformers import OwlViTProcessor, OwlViTForObjectDetection
from PIL import Image
import torch


class ViT():
    def __init__(self, device, threshold=0.1):
        self.device = device
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
        #self.model.eval()
        self.threshold = threshold

    def exist(self, image, query):
        if isinstance(image, str):
            image = Image.open(image)
        query = 'a photo of a ' + query
        inputs = self.processor(text=query, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)
        scores = results[0]["scores"]

        if max(scores) > self.threshold:
            # return "There exist {}.".format(query)
            return str(True)
        else:
            # return "There does not exist {}.".format(query)
            return str(False)


    def expand_bbox(self, bbox, img_width, img_height, scale=0.2):
        # 计算边界框的宽度和高度
        bbox_width = bbox[2] - bbox[0]
        bbox_height = bbox[3] - bbox[1]

        # 计算需要扩展的宽度和高度
        expand_width = int(bbox_width * scale)
        expand_height = int(bbox_height * scale)

        # 计算扩展后的边界框的左上角和右下角坐标
        expand_top_left_x = max(0, bbox[0] - expand_width)
        expand_top_left_y = max(0, bbox[1] - expand_height)
        expand_bottom_right_x = min(img_width, bbox[2] + expand_width)
        expand_bottom_right_y = min(img_height, bbox[3] + expand_height)

        # 返回扩展后的边界框
        return [expand_top_left_x, expand_top_left_y, expand_bottom_right_x, expand_bottom_right_y]


    def locate(self, image, query, count=False):
        if isinstance(image, str):
            image = Image.open(image)

        width, height = image.size
        inputs = self.processor(text='a photo of a ' + query, images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.Tensor([image.size[::-1]])
        results = self.processor.post_process(outputs=outputs, target_sizes=target_sizes)
        boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]

        patch_list = []
        coordinates = []
        scores = []
        for box, score, label in zip(boxes, scores, labels):
            if score >= self.threshold:
                box = [round(i) for i in box.tolist()]
                #expand
                new_box = self.expand_bbox(box, width, height)
                top_left_x, top_left_y, bottom_right_x, bottom_right_y = new_box
                cropped_img = image.crop((top_left_x, top_left_y, bottom_right_x, bottom_right_y))
                patch_list.append(cropped_img)
                coordinates.append(new_box)
                scores.append(score)

        if count:
            # returns all the patches detected
            return {"patches": patch_list, "coordinates": coordinates}
        else:
            #returns the patch with maximum score
            if len(scores) > 0:
                idx = scores.index(max(scores))
                return {"patches": patch_list[idx], "coordinates": coordinates[idx]}
            else:
                return {"patches": None, "coordinates": None}

    def crop(self, image, direction, target=None):
        if isinstance(image, str):
            image = Image.open(image)
        width, height = image.size

        if target:
            coordinates = self.locate(image, target)['coordinates']
        else: coordinates = None

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
            return {"patches": image, "coordinates": None}

        cropped_img = image.crop(new_box)
        return {"patches": cropped_img, "coordinates": new_box}
