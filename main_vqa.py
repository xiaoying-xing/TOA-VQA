import argparse
import json
import os
import pickle
import random
from tqdm import tqdm
from dataset import myDataset
from chatVQA_ok_program_cluster import VQA
from openai_tools import set_openai_key
from utils import evaluation, print_history
import subprocess
import atexit
random.seed(1024)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='datasets/', help='root path to the datasets')
    parser.add_argument('--save_root', type=str, default='experiments/', help='root path for saving results')
    parser.add_argument('--name', type=str, default='test',
                        help='name for this experiment. Results will be saved in save_root/name/dataset/')
    parser.add_argument('--dataset', choices=['GQA', 'OKVQA', 'A-OKVQA', 'VQA_v2'], default='OKVQA',
                        help='Names of the datasets to use in the experiment')
    parser.add_argument('--filename', type=str, required=True, help='name for the saved results file')
    parser.add_argument('--split', choices=['test','test_all','test_balanced','testdev_all','testdev_balanced'], default='test')
    parser.add_argument('--max_num', type=int, default=2000, help='number of data samples to test')
    parser.add_argument('--num_example', type=int, default=16, help='number of prompt examples')
    parser.add_argument('--hypothesis', action='store_false', help='ablation on removing hypothesis in the format')
    parser.add_argument('--ablation', action='store_false', help='ablation on the prompt head')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo', choices=['text-davinci-003', 'text-davinci-002', 'davinci', 'gpt-3.5-turbo', 'llama','anthropics'],
                        help='model used for task scheduler')
    parser.add_argument('--load_caption', type=bool, default=True, help='load the image captioning in the memory')
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--openai_key', required=True, help='openai key for API')

    args = parser.parse_args()
    return args


def main(args):
    openai_key = args.openai_key
    set_openai_key(openai_key)

    # save dir
    save_path = os.path.join(args.save_root, args.name, args.dataset)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_dir = os.path.join(save_path, args.filename)

    curr_i = 0
    pred = []
    chat_hist = []
    while True:
        try:
            if os.path.exists("epoch.txt"):
                with open("epoch.txt", "r") as f:
                    curr_i = int(f.readline())

            print('----------starting from {} -----------'.format(curr_i))

            if os.path.exists(save_dir):
                pred = json.load(open(save_dir))
                print('loaded prediction file')

            dataset = myDataset(args.data_root, args.dataset, args.split, args.max_num, args.load_caption)
            if args.dataset=='OKVQA':
                if args.num_example == 0:
                    test_feat = None
                    example_feat = None
                    example_promt = None
                else:
                    test_feat = pickle.load(open('datasets/OKVQA/clip_feat_test.pkl', 'rb'))
                    example_feat = pickle.load(open('datasets/OKVQA/clip_feat_example.pkl', 'rb'))
                    if args.hypothesis == True:
                        example_promt = json.load(open('example_okvqa/example_cluster_program.json'))
                    else:
                        print('---------Abation of no hypothesis----------')
                        example_promt = json.load(open('example_okvqa/example_cluster_program_nohypo2.json'))
                vqa_agent = VQA(args, planner=args.model, save_path=save_path, examples=example_promt,
                                data_feats=test_feat,
                                example_feats=example_feat, num_example=args.num_example)

            elif args.dataset == 'VQA_v2':
                example_promt = [line.strip() for line in open('example_vqav2/example_cluster_program.txt').readlines()]
                vqa_agent = VQA(args, planner=args.model, save_path=save_path, examples=example_promt)

            for i in tqdm(range(curr_i, dataset.len)):
                keys = [item['question_id'] for item in pred]
                q_id, question, img_id, gt_answer, caption = dataset.get_input(i)
                if q_id in keys:
                    continue
                print('--------------------')
                print(q_id, img_id, 'ground truth:', gt_answer)
                answer, chat = vqa_agent.vqa(question, q_id, img_id, caption)
                #gt.append(gt_answer)
                #chat_hist.append({"question_id":q_id, "chat": chat})

                #chat_history = json.load(open(os.path.join(save_path, 'cluster8_euclidean.json')))
                #chat_history[q_id] = {"chat_history": chat, "prediction": answer, "gt": gt_answer}

                #with open(os.path.join(save_path, 'cluster8_euclidean.json'), 'w') as f:
                #    json.dump(chat_history, f)
                #accuracy = evaluation(prediction, gt)
                #print('epoch:',i,'Overall accuracy:',accuracy)

            #accuracy = evaluation(prediction, gt)
            #print('epoch:', i, 'Overall accuracy:', accuracy)
                pred.append({'question_id': q_id, 'answer': answer})

            with open(save_dir, 'w') as f:
                json.dump(pred, f)
            process.terminate()
            atexit.register(process.terminate)

        except Exception as e:
            print('------------ERROR------------')
            print(e)
            with open("epoch.txt", "w") as f:
                f.write(str(i))

            with open(save_dir, 'w') as f:
                json.dump(pred, f)

            # restart the program when broken
            process = subprocess.Popen(['python', "main_vqa.py"])
            process.terminate()
            atexit.register(process.terminate)


if __name__ == '__main__':
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    args = parse()
    main(args)
