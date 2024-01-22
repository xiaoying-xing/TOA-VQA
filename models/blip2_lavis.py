import json
import pickle

from lavis.models import load_model_and_preprocess
from PIL import Image

class Blip():
    def __init__(self, device, name="blip2_t5"):
        self.device = device
        self.model, self.vis_processors, _ = load_model_and_preprocess(name=name, model_type="pretrain_flant5xxl",
                                                             is_eval=True, device=device)
    def caption(self, image):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        caption = self.model.generate({"image": image})
        return caption

    def simple_query(self, image, q):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        else:
            keys = list(image.keys())
            if len(keys) > 1:
                image = image[keys[-1]]
            else:
                image = image['ori_img']
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        answer = self.model.generate({"image": image, "prompt": "Question: {} Answer:".format(q)})
        if isinstance(answer, list):
            answer = answer[0]
        return answer

    def query_attr(self, image, object, attribute):
        if isinstance(image, str):
            image = Image.open(image).convert('RGB')
        image = self.vis_processors["eval"](image).unsqueeze(0).to(self.device)
        query = 'What is the {} of the {}'.format(attribute, object)
        answer = self.model.generate({"image": image, "prompt": "Question: {} Answer:".format(query)})
        #if 'yes' in answer:
            #return '{} is {}'.format(object, attribute)
            #return "True"
        #elif 'no' in answer:
            #return '{} is not {}'.format(object, attribute)
            #return "False"
        #else:
            #return "Unkown"
        return answer


'''
model = Blip('cpu')
img_path = '../datasets/OKVQA/val2014/COCO_val2014_{}.jpg'.format(str(518721).zfill(12))
print(model.simple_query(img_path, 'What is this animal?'))
print(model.query_attr(img_path, 'animal', 'type'))
print(model.query_attr(img_path, 'tongue', 'length'))
print(model.simple_query(img_path, 'Is the tongue of the giraffe long?'))
print(model.simple_query(img_path, 'Is the tongue of the giraffe 20 inches?'))



img_path = '../datasets/OKVQA/val2014/COCO_val2014_{}.jpg'.format(str(467675).zfill(12))
print(model.simple_query(img_path, 'What is the face the cat is making?'))
print(model.simple_query(img_path, 'Is the cat happy?'))

img_path = '../datasets/OKVQA/val2014/COCO_val2014_{}.jpg'.format(str(499480).zfill(12))
print(model.simple_query(img_path, 'What appliance in this image?'))
'''

'''
import json
from tqdm import tqdm

data = json.load(open('/files0/home/xiaoying/chatVQA/datasets/OKVQA/process_test.json'))
print(len(data))
model = Blip('cpu')
result = []
for line in tqdm(data):
    img = '/files0/home/xiaoying/chatVQA/datasets/OKVQA/val2014/COCO_val2014_{}.jpg'.format(str(line['image_id']).zfill(12))
    caption = model.caption(img)
    line['caption'] = caption
    result.append(line)
with open('/files0/home/xiaoying/chatVQA/datasets/OKVQA/process_test_caption.json','w') as f:
    json.dump(result,f)
'''

'''
ids = ['19776016', '00156022', '16421592', '0529253', '06716575', '11779728', '13568292', '071028324', '15786411', '19692243', '10569062', '00297941', '16312243', '08158827', '111051819', '05709053', '06224368', '12826723', '06203466', '12143165', '05749185', '1873912', '181040668', '03986769', '12212754', '12607036', '12742963', '08816216', '14933831', '01612826', '17773292', '04132006']
model = Blip('cpu')
data = json.load(open('/files0/home/xiaoying/chatVQA/datasets/GQA/val_balanced_questions.json'))

for id in ids:
    img_id = data[id]['imageId']
    img = '/files0/home/xiaoying/chatVQA/datasets/GQA/images/{}.jpg'.format(img_id)
    caption = model.caption(img)
    print(id, data[id]['question'],data[id]['answer'], caption)
'''
