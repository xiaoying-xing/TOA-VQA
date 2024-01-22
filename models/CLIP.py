import torch
import clip
import numpy as np
import json
import pickle
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

data = json.load(open('../datasets/OKVQA/process_test.json'))
for value in data:
    if value['question'] == 'What design is on the headboard of the bed?':
        print(value)

'''
result = {}
for key, value in data.items():
    for line in annotation:
        if line['question_id'] == key:
            print(line)
'''

'''
#test_dev = json.load(open('/files0/home/xiaoying/chatVQA/datasets/GQA/testdev_balanced_questions.json'))
#val_data = json.load(open('/files0/home/xiaoying/chatVQA/datasets/GQA/val_balanced_questions.json'))
process_test = json.load(open('/files0/home/xiaoying/chatVQA/datasets/A-OKVQA/process_test.json'))

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

features = {}
#ids = np.array(list(process_test.keys()))
for line in process_test:
    q = line['question']
    q_id = line['question_id']
    q = clip.tokenize(q).to(device)
    with torch.no_grad():
        text_features = model.encode_text(q).cpu().numpy()
        features[q_id] = np.squeeze(text_features)

with open('../datasets/A-OKVQA/clip_feat_test.pkl','wb') as f:
    pickle.dump(features, f)

'''

'''
#val_data = json.load(open('/files0/home/xiaoying/chatVQA/datasets/GQA/val_balanced_questions.json'))
#val_ids = np.array(list(val_data.keys()))

#test_data = json.load(open('../datasets/OKVQA/process_test.json'))
test_features = pickle.load(open('../datasets/A-OKVQA/clip_feat_val.pkl','rb'))
test_ids = np.array(list(test_features.keys()))
#val_features = pickle.load(open('clip_feat_val.pkl','rb'))
test_features = np.stack(list(test_features.values())).reshape(len(test_ids),-1)
#val_features = np.array(val_features)

clustering = AgglomerativeClustering(n_clusters=16, linkage='average', affinity='cosine')
clustering.fit(test_features)

representatives = []
representatives_ids = []
for i in range(clustering.n_clusters):
    mask = clustering.labels_ == i
    mask = list(np.nonzero(mask)[0])
    cluster_data = test_features[mask]
    cluster_ids = test_ids[mask]  # 假设原始ID存储在original_ids变量中

    dist_to_center = []
    center = np.mean(cluster_data, axis=0)
    for data in cluster_data:
        dist_to_center.append(cosine_similarity(data.reshape(1,-1), center.reshape(1,-1)))
    #dist_to_center = np.sum((cluster_data - clustering.cluster_centers_[i]) ** 2, axis=1)
    #dist_to_center = np.sum((val_features - kmeans.cluster_centers_[i]) ** 2, axis=1)
    #dist_to_center = cosine_distance(val_features[0], kmeans.cluster_centers_[i])
    representative_index = np.argmax(dist_to_center)
    representatives_ids.append(cluster_ids[representative_index])
    representatives.append(cluster_data[representative_index])
    #representatives_ids.append(cluster_ids[representative_index])

print(representatives_ids)
result = {}
assert len(representatives_ids) == len(representatives)
for id, feat in zip(representatives_ids, representatives):
    result[id] = feat

with open('../datasets/A-OKVQA/clip_feat_example.pkl','wb') as f:
    pickle.dump(result, f)

'''

'''
from PromptCAP import prompt_cap
model = prompt_cap('cpu')

ids = [3433155, 4676755, 2100995, 4011575, 825515, 5187215, 2290965, 3859185, 1485085, 3525495, 28905, 921345, 5249545, 1828345, 2594655, 3733445, 3085045, 671255, 3719995, 4642965, 2043815, 4371265, 128875, 3494865, 5058495, 5084405, 4994805, 5518115, 4974665, 2451535, 2336605, 575505]
test_data = json.load(open('../datasets/OKVQA/process_test.json'))
for line in test_data:
    if line['question_id'] in ids:
        id = line['image_id']
        img_path = '/files0/home/xiaoying/chatVQA/datasets/OKVQA/val2014/COCO_val2014_{}.jpg'.format(str(id).zfill(12))
        question = line['question']
        caption = model.q_caption(img_path, question)
        print(line)
        print(caption)
        print('-----------')

'''
'''
my_list = [line.strip() for line in open('../example_aokvqa/example_cluster_program.txt').readlines()]
#ids = ['19776016', '00156022', '16421592', '0529253', '06716575', '11779728', '13568292', '071028324', '15786411', '19692243', '10569062', '00297941', '16312243', '08158827', '111051819', '05709053', '06224368', '12826723', '06203466', '12143165', '05749185', '1873912', '181040668', '03986769', '12212754', '12607036', '12742963', '08816216', '14933831', '01612826', '17773292', '04132006']
ids = list(example.keys())
print(ids)
print(len(ids))
output_list = []
current_sublist = []

# 遍历原始列表
for item in my_list:
    # 如果当前项目不是空字符串，则将其添加到当前子列表中
    if item != '':
        current_sublist.append(item)
    # 如果当前项目是空字符串，且当前子列表不为空，则将当前子列表添加到输出列表中，并开始一个新的子列表
    elif current_sublist != []:
        output_list.append(current_sublist)
        current_sublist = []

# 如果循环结束后当前子列表不为空，则将其添加到输出列表中
if current_sublist != []:
    output_list.append(current_sublist)

output = {}
assert len(ids) == len(output_list)
for i in range(len(ids)):
    output[str(ids[i])] = output_list[i]

print(output)
with open('../example_aokvqa/example_cluster_program.json', 'w') as f:
    json.dump(output, f)
'''

