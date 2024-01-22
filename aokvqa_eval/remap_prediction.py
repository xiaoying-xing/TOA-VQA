import argparse
import pathlib
import json
from tqdm import tqdm
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

def post_process():
    pred = json.load(open('/files0/home/xiaoying/chatVQA/experiments/test/OKVQA/result_prog_cluster16.json'))
    print(len(pred))
    num_to_word = {
        '0': "zero", '1': "one", '2': "two", '3': "three", '4': "four",
        '5': "five", '6': "six", '7': "seven", '8': "eight", '9': "nine"
    }

    to_be_removed = {'a', 'an', 'the', 'to', ''}

    result = []
    for _line in pred:
        line = _line.copy()
        answer = line['answer']
        answer_list = answer.split(' ')
        answer_list = [item for item in answer_list if item not in to_be_removed]
        answer = ' '.join(answer_list)
        #if answer in num_to_word.keys():
        #    line['answer'] = num_to_word[answer]
        line['answer'] = answer.lower()
        result.append(line)

    with open('/files0/home/xiaoying/chatVQA/experiments/test/OKVQA/result_prog_cluster16_process.json', 'w') as f:
        json.dump(result, f)

post_process()
def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    return dataset


def map_to_choices(dataset, predictions, device='cpu'):
    if isinstance(dataset, list):
        dataset = { dataset[i]['question_id'] : dataset[i] for i in range(len(dataset)) }
    predictions = {item['question_id']:item['answer'] for item in predictions}

    if all([p in dataset[q]['choices'] for q, p in predictions.items()]):
        return predictions

    model = SentenceTransformer('sentence-transformers/average_word_embeddings_glove.6B.300d')
    model.to(device)
    for q in tqdm(predictions.keys()):
        choices = dataset[q]['choices']
        if predictions[q] not in choices:
            for choice in choices:
                if choice in predictions[q] or predictions[q] in choice:
                    predictions[q] = choice
        if predictions[q] not in choices:
            print(predictions[q], choices, dataset[q]['correct_choice_idx'])
            choice_embeddings = model.encode([predictions[q]] + choices, convert_to_tensor=True)
            a_idx = cos_sim(choice_embeddings[0], choice_embeddings[1:]).argmax().item()
            predictions[q] = choices[a_idx]

    return predictions

'''
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aokvqa-dir', type=pathlib.Path, required=True, dest='aokvqa_dir')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--pred', type=argparse.FileType('r'), required=True, dest='prediction_file')
    parser.add_argument('--out', type=argparse.FileType('w'), required=True, dest='output_file')
    args = parser.parse_args()

    dataset = load_aokvqa(args.aokvqa_dir, args.split)
    #predictions = json.load(args.prediction_file)
    predictions = json.load(open('/files0/home/xiaoying/chatVQA/experiments/test/A-OKVQA/result_prog_cluster16_mc2_process.json'))
    predictions = map_to_choices(dataset, predictions)

    json.dump(predictions, args.output_file)
'''
