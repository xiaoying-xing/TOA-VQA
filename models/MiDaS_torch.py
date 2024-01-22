import torch
from torch import hub

class DepthEstimationModel():
    name = 'depth'

    def __init__(self, gpu_number=0, model_type='DPT_Large'):
        self.dev = 'cpu'
        depth_estimation_model = hub.load('intel-isl/MiDaS', model_type, pretrained=True).to(self.dev)
        depth_estimation_model.eval()

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

        if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

        self.depth_estimation_model = depth_estimation_model

    @torch.no_grad()
    def forward(self, image):
        """Estimate depth map"""
        image_numpy = image.cpu().permute(1, 2, 0).numpy() * 255
        input_batch = self.transform(image_numpy).to(self.dev)
        prediction = self.depth_estimation_model(input_batch)
        # Resize to original size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image_numpy.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        # We compute the inverse because the model returns inverse depth
        to_return = 1 / prediction
        to_return = to_return.cpu()
        return to_return  # To save: plt.imsave(path_save, prediction.cpu().numpy())

from PIL import Image
import requests
import torchvision.transforms as transforms
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
image = transforms.ToTensor()(image)
model = DepthEstimationModel()
result = model(image)