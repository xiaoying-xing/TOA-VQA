import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai_tools import *
from models.blip2_lavis import Blip
from models.OWL_ViT import ViT
from models.xvlm.xvlm import XVLMModel
import torch
import os
from models.DINO import DINO
from PIL import Image

vtools = ['exist', 'locate', 'crop', 'crop_image', 'query_attribute', 'filter', 'simple_query', 'count']
chatgpt_models = ['gpt-3.5-turbo']
gpt3_models = ['text-davinci-003', 'text-davinci-002', 'davinci']
#EXAMPLES = [line.strip() for line in open('example_okvqa/example_cluster_program.txt').readlines()]

from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

anthropic = Anthropic(
    api_key="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",
)


class VQA():
    def __init__(self, args, planner="gpt-3.5-turbo", save_path='', num_example = 16,
                 examples = None, data_feats=None, example_feats=None):
        self.data_root = os.path.join(args.data_root, args.dataset)
        self.planner = planner
        self.save_path = save_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() and args.use_gpu else 'cpu')
        self.xvlm = XVLMModel(self.device)
        self.dino = DINO()
        self.blip = Blip(self.device)

        self.messages = []
        self.num_example = num_example
        self.examples = examples
        self.data_feats = data_feats
        self.example_feats = example_feats
        if num_example!=0:
            self.QUESTION_INSTRUCTION = open('example_okvqa/prompt_program.txt').read().strip()
        elif num_example==0:
            self.QUESTION_INSTRUCTION = open('example_okvqa/prompt_program_zeroshot.txt').read().strip()


    '''
    def parse_query(self, query_ori):
        if 'The answer is:' in query_ori:
            query = query_ori.split('The answer is: ')[-1]
            return {'query': query_ori, 'answer':query}

        elif 'the answer is:' in query_ori:
            query = query_ori.split('the answer is: ')[-1]
            return {'query': query_ori, 'answer':query}

        else:
            query = query_ori.split('Next step: ')[-1]
            # parse query
            if query[-1] == '?':
                query = 'simple_query({})'.format(query)
            left = query.find('(')
            right = query.find(')')
            mid = query.find(',')
            function = query[:left]
            obj, prop = '',''
            if mid==-1:
                obj = query[left+1:right]
                obj = obj[obj.find('=') + 1:]
            else:
                obj = query[left+1:mid]
                obj = obj[obj.find('=') + 1:]
                prop = query[mid+2:right]
                prop = prop[prop.find('=') + 1:]

            query_parsed = {
                "query": query_ori,
                "function": function,
                "object": obj,
                "property": prop
            }
            return query_parsed
    '''

    def parse_query(self, query_ori):
        invalid_tokens = ['cannot determine', 'cant determine', 'Sorry', 'sorry', 'cannot answer',
                          'Cannot determine', 'Cannot answer', 'Cannot be determined', 'cannot be determined']
        if 'Verification:' not in query_ori or 'Answer:' not in query_ori:
            return {
                "query": query_ori,
                "function": 'base VQA'
            }
        temp = query_ori.split('Verification: ')[1]
        query = temp.split('Answer: ')[0].strip()
        answer = temp.split('Answer: ')[1].strip()

        if query == 'None' and (answer == 'None' or any(token in answer for token in invalid_tokens)):
            return {
                "query": query_ori,
                "function": 'base VQA'
            }

        elif query == 'None' and answer != 'None':
            return {'query': query_ori, 'answer': answer}

        else:
            if any(query in item['verification'] for item in self.query_history):
                return {
                    "query": query_ori,
                    "function": 'base VQA'
                }

            else:
                left = query.find('(')
                right = query.find(')')
                mid = query.find(',')
                function = query[:left]
                obj, prop = '', ''
                if mid == -1:
                    obj = query[left + 1:right]
                    obj = obj[obj.find('=') + 1:]
                else:
                    obj = query[left + 1:mid]
                    obj = obj[obj.find('=') + 1:]
                    prop = query[mid + 2:right]
                    prop = prop[prop.find('=') + 1:]

                if function not in vtools:
                    return {"query": query_ori,
                            "function": 'base VQA'}
                query_parsed = {
                    "query": query_ori,
                    "verification": query,
                    "function": function,
                    "object": obj,
                    "property": prop
                }
                return query_parsed

    def program_generation(self, question, caption):
        if self.planner in chatgpt_models:
            chatgpt_messages = self.all_messages + self.messages
            query, n_tokens = call_chatgpt(chatgpt_messages)
        elif self.planner in gpt3_models:
            gpt3_messages = self.all_messages + self.messages
            new_message = []
            for line in gpt3_messages:
                if 'Question:' in line['content'] or 'Caption:' in line['content'] or line['role']=='system':
                    str_message = line['content']
                else:
                    str_message = line['role'] + ': ' + line['content']
                new_message.append(str_message)
            new_message = '\n'.join(new_message)
            query, n_tokens = call_gpt3(new_message)

        elif self.planner  == 'anthropics':
            messages = self.all_messages + self.messages
            new_message = ''
            for line in messages:
                if line['role'] == 'system' or line['role'] =='user':
                    new_message = new_message + HUMAN_PROMPT + ' ' + line['content'] + ' '
                else:
                    new_message = new_message + AI_PROMPT + ' ' + line['content']
            new_message += AI_PROMPT
            completion = anthropic.completions.create(
                model="claude-2",
                max_tokens_to_sample=300,
                prompt=new_message,
            )
            query = completion.completion
            n_tokens = 0

        elif self.planner == 'llama':
            query = input("------input llama query-------")
            n_tokens = 0

        else: raise ValueError('Invalid LLM usage')

        query = query.replace("'", "").replace('"', '').strip()
        query = query.replace('\n','')
        #query = query.split('\n')[0].replace('assistant: ','')
        #query = "Reasoning:" + query.split('Reasoning:')[1]
        print('GPT:', query)
        step = self.parse_query(query)
        self.query_history.append(step)

        self.messages.append({"role": "assistant", "content": query})

        return step, n_tokens

    def program_executor(self, query, question=None, img=None):
        if query['function'] == 'description':
            response = self.promptcap.q_caption(self.img_path, question)
        # elif query['function'] == 'objects':
        #    response = self.faster_rcnn.detect(self.image_patches['ori_img'])
        elif query['function'] == 'exist':
            response = self.dino.exist(self.image_patches['ori_img'], query['object'])
        elif query['function'] == 'query_attribute':
            # response = self.xvlm.query_attribute(img_path, query[left+1:mid], query[mid+2:right])
            obj_name = query['object']
            if obj_name in self.image_patches.keys():
                img_patch = self.image_patches[obj_name]
            else:
                img_patch = self.image_patches['ori_img']
            response = self.blip.query_attr(img_patch, obj_name, query['property'])

        elif query['function'] == 'filter':
            temp = query['property']
            prop = temp[temp.find(':') + 1:]
            obj_name = query['object']
            if obj_name in self.image_patches.keys():
                img_patch = self.image_patches[obj_name]
            else:
                # add locate function?
                img_patch = self.image_patches['ori_img']
            response = self.xvlm.filter(img_patch, obj_name, prop)

        elif query['function'] == 'simple_query':
            response = self.blip.simple_query(self.image_patches, query['object'])

        elif query['function'] == 'count':
            response = self.dino.locate(self.image_patches['ori_img'], query['object'], count=True)

        elif query['function'] == 'locate':
            cropped, box = self.dino.locate(self.image_patches['ori_img'], query['object'])
            if box:
                self.image_patches[query['object']] = cropped
                response = 'Located {}'.format(query['object'])
            else:
                response = 'Located {}'.format(query['object'])

        elif query['function'] == 'crop':
            self.image_patches['ori_img'] = self.dino.crop(self.image_patches['ori_img'], query['property'],
                                                           query['object'])
            response = 'Cropped image'

        elif query['function'] == 'crop_image':
            self.image_patches['ori_img'] = self.dino.crop(self.image_patches['ori_img'], query['object'])
            response = 'Cropped image'
        else:
            print(query)
            raise ValueError("function not defined.")

        if isinstance(response, list):
            response = response[0]

        print('Vision:',response)

        return response

    def vqa(self, question, question_id, img_id=None, caption=None, choices=None):
        self.query_history = []
        self.all_messages = [{"role": "system", "content": self.QUESTION_INSTRUCTION}]
        self.messages = [{"role": "user", "content": 'Question: ' + question}]
        if choices:
            self.messages.append({"role": "user", "content": 'Choices: ' + str(choices).replace('\'','')})
        self.messages.append({"role": "user", "content": 'Caption: ' + caption})

        if self.examples and self.num_example != 0:
            # select prompting examples
            data_feat = self.data_feats[question_id]
            example_feat = np.array(list(self.example_feats.values()))
            similarity = cosine_similarity(data_feat.reshape(1, -1), example_feat)
            similarity = np.squeeze(similarity)
            indices = np.argsort(-similarity)[:self.num_example]
            example_keys = [list(self.example_feats.keys())[i] for i in indices]
            
            for key in example_keys:
                example = self.examples[str(key)]
                for line in example:
                   if 'Question' in line or 'Caption' in line:
                       self.all_messages.append({'role': 'user', 'content': line})
                   elif 'AI' in line:
                       self.all_messages.append({'role': 'assistant', 'content': line[4:]})
                   elif 'Human' in line:
                       self.all_messages.append({'role': 'user', 'content': line[7:]})
            '''
            for line in self.examples:
                if 'Question' in line or 'Caption' in line or 'Choices' in line:
                    self.all_messages.append({'role': 'user', 'content': line})
                elif 'AI' in line:
                    self.all_messages.append({'role': 'assistant', 'content': line[4:]})
                elif 'Human' in line:
                    self.all_messages.append({'role': 'user', 'content': line[7:]})
            '''

        total_token = 0
        self.img_path = os.path.join('datasets/OKVQA', 'val2014', 'COCO_val2014_{}.jpg'.format(str(img_id).zfill(12)))
        self.image_patches = {}
        self.image_patches["ori_img"] = Image.open(self.img_path).convert('RGB')
        #caption = self.promptcap.q_caption(self.img_path, question)
        print('Question:',question+'and'+'Caption:', caption)
        #print('Choices:', choices)
        for k in range(8):
            step, n_token = self.program_generation(question, caption)
            total_token += n_token
            if 'answer' in step.keys():
                answer = step['answer']
                #self.print_history(question, img_id)
                return answer, self.messages
            elif step['function'] == 'base VQA':
                answer = self.blip.simple_query(self.img_path, question)
                print('base VQA:',answer)
                self.messages.append({"role": "base VQA", "content": answer})
                #self.print_history(question, img_id)
                return 'baseVQA_'+answer, self.messages
            else:
                response = self.program_executor(step, img_id)
                self.messages.append({"role": "user", "content": response})

        #self.print_history(question, img_id)
        # question answering agent
        print('Did not get answer within 8 rounds.')
        return 'NA', self.messages
        # answer = self.answer(question)
        # print('Answer:', answer)
        # return answer
