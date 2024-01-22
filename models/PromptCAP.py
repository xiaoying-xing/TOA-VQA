import json

import torch
from promptcap import PromptCap
from PIL import Image


class prompt_cap():
    def __init__(self, device):
        self.model = PromptCap(
            "vqascore/promptcap-coco-vqa")  # .to(device)  # also support OFA checkpoints. e.g. "OFA-Sys/ofa-large"
        self.model.cuda()
        self.prompt_head = "please describe this image according to the given question: "

    def q_caption(self, image, question):
        prompt = self.prompt_head + question
        return self.model.caption(prompt, image)

    def g_caption(self, image):
        prompt = "what does the image describe?"
        return self.model.caption(prompt, image)
