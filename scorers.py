from collections import defaultdict
import numpy as np
from tqdm import tqdm
import concurrent.futures


def predict_on_example(inputs):
    ex, predictor, prompt = inputs
    img_path = ex['img_path']
    pred = predictor.inference(prompt, [img_path])
    return prompt, ex, pred


def predict_on_example_attr(inputs):
    ex, predictor, pred_prompt, prompt, attr = inputs
    img_path = ex['img_path']
    pred = predictor.inference(pred_prompt, [img_path], attr=attr)
    return prompt, ex, pred


class Cached01Scorer:

    def __init__(self):
        self.cache = {}

    def __call__(self, predictor, prompts, data, pred_prompt=None, attribute_cache=None, agg='mean', max_threads=1):
        def compute_scores(prompts_exs):
            out_scores = {}
            if attribute_cache != None:
                inputs = [(ex, predictor, pred_prompt, prompt, attribute_cache[f'{prompt}'][f'{ex}']) for prompt, ex in prompts_exs]
            else:
                inputs = [(ex, predictor, prompt) for prompt, ex in prompts_exs]
            with concurrent.futures.ProcessPoolExecutor(max_workers=max_threads) as executor:
                if attribute_cache != None:
                    futures = [executor.submit(predict_on_example_attr, ex) for ex in inputs]
                else:
                    futures = [executor.submit(predict_on_example, ex) for ex in inputs]
                for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='01 scorer'):
                    prompt, ex, pred = future.result()
                    if pred == ex['label']:
                        out_scores[f'{ex}-{prompt}'] = 1
                    else:
                        out_scores[f'{ex}-{prompt}'] = 0
            return out_scores

        cached_scores = defaultdict(list)
        prompts_exs_to_compute = []
        for ex, prompt in [(ex, prompt) for ex in data for prompt in prompts]:
            if f'{ex}-{prompt}' in self.cache:
                cached_scores[prompt].append(self.cache[f'{ex}-{prompt}'])
            else:
                prompts_exs_to_compute.append((prompt, ex))
        computed_scores = compute_scores(prompts_exs_to_compute)
        for prompt, ex in prompts_exs_to_compute:
            self.cache[f'{ex}-{prompt}'] = computed_scores[f'{ex}-{prompt}']
            cached_scores[prompt].append(computed_scores[f'{ex}-{prompt}'])

        if agg == 'mean':
            return [np.mean(cached_scores[prompt]) for prompt in prompts]
        else:
            raise Exception('Unk agg: ' + agg)
