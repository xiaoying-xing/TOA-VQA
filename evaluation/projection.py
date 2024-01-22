import json
import numpy as np
import clip
import torch
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

'''
def prepare_pool():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    annotations = '/files0/home/xiaoying/chatVQA/datasets/OKVQA/mscoco_val2014_annotations.json'
    annotations = json.load(open(annotations))['annotations']
    pool = []
    for line in tqdm(annotations):
        gt = line['answers']
        _gt = set([item['answer'] for item in gt])
        _gt = list(_gt)
        pool += _gt

    print(len(pool))
    output = {}
    for word in tqdm(pool):
        word_token = clip.tokenize(word).to(device)
        with torch.no_grad():
            text_features = model.encode_text(word_token).cpu().numpy()
            output[word] = np.squeeze(text_features)

    with open('/files0/home/xiaoying/chatVQA/datasets/OKVQA/answer_pool_clip.pkl','wb') as f:
        pickle.dump(output, f)
'''

'''
def prepare_pool_small():
    pool = json.load(open('/files0/home/xiaoying/prophet/assets/answer_dict_okvqa.json'))
    output = {}
    for word in tqdm(pool):
        word_token = clip.tokenize(word).to(device)
        with torch.no_grad():
            text_features = model.encode_text(word_token).cpu().numpy()
            output[word] = np.squeeze(text_features)

    with open('/files0/home/xiaoying/chatVQA/datasets/OKVQA/answer_pool_clip_small.pkl', 'wb') as f:
        pickle.dump(output, f)
'''


def map_to_choices(predictions, save_path, device='cpu'):
    #model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
    #model.to(device)

    predictions = json.load(open(predictions))
    annotations = '/files0/home/xiaoying/chatVQA/datasets/OKVQA/my_annotations.json'
    annotations = json.load(open(annotations))['annotations']
    pool = []
    for line in annotations:
        gt = line['answers']
        _gt = set([item['answer'] for item in gt])
        _gt = list(_gt)
        pool += _gt

    pool = list(set(pool))

    print(len(pool))
    result = []
    for line_pred in predictions:
        pred = line_pred['answer']
        if pred in pool:
            continue

        pred_set = set(pred.split())
        for item in pool:
            item_set = set(item.split())
            if pred_set.intersection(item_set):
                print(pred, '|', item)

        #embedding = model.encode([pred] + pool, convert_to_tensor=True)
        #a_idx = cos_sim(embedding[0], embedding[1:]).argmax().item()
        #line_pred['answer'] = pool[a_idx]
        #result.append(line_pred)

    #with open(save_path, 'w') as f:
    #    json.dump(result, f)


def project_answer_clip(pred_path, save_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    answer_pool_clip = pickle.load(open('/files0/home/xiaoying/chatVQA/datasets/OKVQA/answer_pool_clip_small.pkl','rb'))
    pool_keys = list(answer_pool_clip.keys())
    pool_features = list(answer_pool_clip.values())
    pool_features = np.array(pool_features)

    pred_file = json.load(open(pred_path))

    processed_pred = []
    for line in tqdm(pred_file):
        pred = line['answer']
        word = clip.tokenize(pred).to(device)
        with torch.no_grad():
            text_features = model.encode_text(word).cpu().numpy()
            #text_features = np.squeeze(text_features)

        #similarity = []
        #for pool_feat in pool_features:
        #    pool_feat = pool_feat.reshape(1, -1)
        #    similarity.append(cosine_similarity(text_features, pool_feat))

        similarity = cosine_similarity(text_features, pool_features)
        similarity = np.squeeze(similarity)

        idx = np.argmax(similarity)
        result = pool_keys[idx]
        line['answer'] = result
        processed_pred.append(line)

    with open(save_path, 'w') as f:
        json.dump(processed_pred, f)


#prepare_pool()
#project_answer('/files0/home/xiaoying/chatVQA/experiments/test/OKVQA/result_prog_cluster16.json', '/files0/home/xiaoying/chatVQA/experiments/test/OKVQA/result_prog_cluster16_processed.json')
map_to_choices('/files0/home/xiaoying/chatVQA/experiments/test/OKVQA/result_prog_cluster16_process.json', '/files0/home/xiaoying/chatVQA/experiments/test/OKVQA/result_prog_cluster16_processed.json')