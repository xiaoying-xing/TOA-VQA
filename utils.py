import json
import os
import sys

'''
class HiddenPrints:
    def __init__(self, model_name=None, console=None, use_newline=True):
        self.model_name = model_name
        self.console = console
        self.use_newline = use_newline
        self.tqdm_aux = None

    def __enter__(self):
        import tqdm  # We need to do an extra step to hide tqdm outputs. Does not work in Jupyter Notebooks.

        def nop(it, *a, **k):
            return it

        self.tqdm_aux = tqdm.tqdm
        tqdm.tqdm = nop

        if self.model_name is not None:
            self.console.print(f'Loading {self.model_name}...')
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        # May not be what we always want, but some annoying warnings end up to stderr
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout
        sys.stdout = self._original_stderr
        if self.model_name is not None:
            self.console.print(f'{self.model_name} loaded ')
        import tqdm
        tqdm.tqdm = self.tqdm_aux
'''


def post_process(prediction):
    if isinstance(prediction, list):
        prediction = prediction[0] if len(prediction) > 0 else "no"
    if prediction is None or prediction == 'cannot be determined' or prediction == 'n/a':
        prediction = "no"

    prediction = prediction.replace('.', '')
    prediction = prediction.strip().lower()

    if prediction.startswith("a "):
        prediction.replace('a ', '')
    elif prediction.startswith('the '):
        prediction.replace('the ', '')
    elif prediction.startswith("yes"):
        prediction = 'yes'
    elif prediction.startswith("no"):
        prediction = 'no'

    if 'yes' in prediction or 'true' in prediction:
        prediction = 'yes'
    elif 'no' in prediction or 'false' in prediction or 'not' in prediction:
        prediction = 'no'

    if len(prediction.split()) > 5:
        prediction = 'no'
    return prediction


def evaluation(prediction, gt_answer):
    if len(prediction) == 0:  # if no prediction, return 0
        return 0
    assert len(prediction) == len(gt_answer)
    pred_gt_filtered = [(pred, gt) for pred, gt in zip(prediction, gt_answer) if gt != '']
    score = 0
    for p, g in pred_gt_filtered:
        if g in post_process(p):
            score += 1
    return score / len(pred_gt_filtered)


def print_history(messages):
    for item in messages:
        print(item["role"], ':', item["content"])
