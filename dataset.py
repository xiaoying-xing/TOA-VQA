import random
import os
import json
from random import sample
random.seed(1024)

class myDataset():
    def __init__(self, data_root, dataset_name, split, max_num=None, load_caption=False, multi_choice=False):
        self.dataset_name = dataset_name
        self.load_caption = load_caption
        self.multi_choice = multi_choice
        data_path = os.path.join(data_root, dataset_name, 'process_{}_cap.json'.format(split))
        self.data = json.load(open(data_path))

        if dataset_name == 'GQA':
            if max_num and max_num < len(self.data):
                selected_ids = sample(list(self.data.keys()), max_num)
                self.data = {idx: self.data[idx] for idx in selected_ids}
            self.q_ids = list(self.data.keys())
            self.len = len(self.q_ids)

        else:
            if max_num and max_num < len(self.data):
                self.data = sample(self.data, max_num)
            self.len = len(self.data)

    def get_input(self, index):
        if self.dataset_name == 'GQA':
            q_id = self.q_ids[index]
            data = self.data[q_id]
            question = data['question']
            gt_answer = data['answer']
            img_id = data['imageId']
            if self.load_caption:
                caption = data['caption']
            else:
                caption = None
            return q_id, question, img_id, gt_answer, caption

        elif self.dataset_name == 'OKVQA' or self.dataset_name=='VQA_v2':
            data = self.data[index]
            q_id = data['question_id']
            question = data['question']
            img_id = data['image_id']
            gt_answer = data['answers']
            if self.load_caption:
                caption = data['caption']
            else:
                caption = None
            return q_id, question, img_id, gt_answer, caption

        elif self.dataset_name == 'A-OKVQA':
            data = self.data[index]
            q_id = data['question_id']
            question = data['question']
            img_id = data['image_id']
            #gt_answer = self.data[index]['direct_answers']
            gt_answer = data['choices'][data['correct_choice_idx']]
            if self.load_caption:
                caption = data['caption']
            else:
                caption = None
            if self.multi_choice:
                choices = data['choices']
                return q_id, question, img_id, gt_answer, caption, choices
            else: return q_id, question, img_id, gt_answer, caption




'''
question = json.load(open('datasets/OKVQA/OpenEnded_mscoco_val2014_questions.json'))['questions']
annotation = json.load(open('datasets/OKVQA/mscoco_val2014_annotations.json'))['annotations']
result = []
for i in range(len(question)):
    q = question[i]
    an = annotation[i]
    assert q['question_id'] == an['question_id']
    q.update(an)
    result.append(q)

with open('datasets/OKVQA/process_test.json','w') as f:
    json.dump(result, f)
'''