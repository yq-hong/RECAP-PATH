import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from tqdm import tqdm
import time
import datetime
import json
import argparse
import utils
import tasks
import predictors
import optimizers
import generator
import get_keywords


def get_task_class(args):
    if args.task == 'BRACS':
        return tasks.BRACSBinaryTask(args.data_dir, args.max_threads, args.class0, args.class1)
    elif args.task == 'BACH':
        return tasks.BACHBinaryTask(
            args.data_dir, args.max_threads, args.class0, args.class1
        )
    elif args.task == 'SICAPv2':
        return tasks.SICAPv2BinaryTask(
            args.data_dir, args.max_threads, args.class0, args.class1
        )
    elif args.task == 'BRACS_multi':
        return tasks.BRACSMultiTask(args.data_dir, args.max_threads)
    else:
        raise Exception(f'Unsupported task: {args.task}')


def get_predictor(configs):
    if (
        configs['task'] == 'BRACS'
        or configs['task'] == 'BACH'
        or configs['task'] == 'SICAPv2'
       ):
        return predictors.TwoClassPredictor(configs)
    elif configs['task'] == 'BRACS_multi':
        return predictors.MultiClassPredictor(configs)
    else:
        raise Exception(f"Unsupported task: {configs['task']}")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='BRACS', choices=['BRACS', 'BRACS_multi', 'BACH', 'SICAPv2'])
    parser.add_argument('--class0', default='N')
    parser.add_argument('--class1', default='IC')
    parser.add_argument('--model', default='gemini', choices=['gemini', 'gpt4o'])
    parser.add_argument('--gradient_model', default='gemini')
    parser.add_argument('--data_dir', default='../file_names')

    parser.add_argument('--out_num', default='0')
    parser.add_argument('--max_threads', default=16, type=int)
    parser.add_argument('--max_token', default=1024, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)

    parser.add_argument('--optimizer', default='nl-gradient')
    parser.add_argument('--rounds', default=6, type=int)
    parser.add_argument('--beam_size', default=4, type=int)
    parser.add_argument('--n_test', default=10, type=int)

    parser.add_argument('--minibatch_size', default=24, type=int)
    parser.add_argument('--n_gradients', default=3, type=int, help='# generated gradients per prompt')
    parser.add_argument('--errors_per_gradient', default=4, type=int,
                        help='# error examples used to generate one gradient')
    parser.add_argument('--gradients_per_error', default=1, type=int, help='# gradient reasons per error')
    parser.add_argument('--steps_per_gradient', default=1, type=int, help='# new prompts per gradient reason')
    parser.add_argument('--mc_samples_per_step', default=1, type=int, help='# synonyms')
    parser.add_argument('--max_expansion_factor', default=5, type=int, help='maximum # prompts after expansion')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if 'multi' in args.task:
        args.out = f'results/{args.out_num}_{args.task}_{args.model}_test_out.txt'
    else:
        args.out = f'results/{args.out_num}_{args.task}_{args.model}_{args.class0}_{args.class1}_test_out.txt'

    configs = vars(args)
    if "gemini" in args.model:
        utils.clear_gemini_img_files(True)

    task = get_task_class(args)
    gpt4 = get_predictor(configs)
    gpt_generator = generator.AttrGredictor(configs)
    optimizer = optimizers.ProTeGi(configs, args.max_threads)

    if os.path.exists(args.out):
        os.remove(args.out)
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')

    if 'multi' in args.task:
        test_exs = task.get_even_exs('test', args.n_test)
    else:
        with open(f'../file_names/test_exs_{args.class0}_{args.class1}.json', 'r') as json_file:
            test_exs = json.load(json_file)

    candidates = [open(f'../prompts/{args.task}_generate.md').read()]
    pred_prompt = open(f'../prompts/{args.task}.md').read()
    with open(args.out, 'a') as outf:
        outf.write(f'pred_prompt-------------------------\n')
        outf.write(f'{pred_prompt}\n\n')

    keyword_pool, test_attr_cache = {}, {}
    for prompt in candidates:
        keyword_pool[f'{prompt}'] = get_keywords.get_text_keywords(prompt)
    prompt_diversity, _, _, _ = get_keywords.get_diversity_score(keyword_pool)

    for round in tqdm(range(configs['rounds'] + 1)):
        print("STARTING ROUND ", round)
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
        start = time.time()

        if round > 0:
            candidates = optimizer.expand_candidates(candidates, prompt_diversity)
            for prompt in candidates:
                keyword_pool[f'{prompt}'] = get_keywords.get_text_keywords(prompt)
            prompt_diversity, _, _, _ = get_keywords.get_diversity_score(keyword_pool)

        candidates = [key for key, _ in sorted(prompt_diversity.items(), key=lambda item: item[1], reverse=True)[:configs['beam_size']]]
        scores = [score for _, score in sorted(prompt_diversity.items(), key=lambda item: item[1], reverse=True)[:configs['beam_size']]]

        with open(args.out, 'a') as outf:
            outf.write(f'{time.time() - start}\n')
            for c in candidates:
                outf.write(json.dumps(c) + '\n')
            outf.write(f'{scores}\n')

        metrics = []
        for prompt in candidates:
            test_attr_cache[f'{prompt}'] = {}
            test_attr_cache = generator.parallel_generate(gpt_generator, prompt, test_exs, test_attr_cache, args.max_threads)

        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds, attr = task.evaluate(gpt4, candidate, test_exs, pred_prompt=pred_prompt, attribute_cache=test_attr_cache)
            metrics.append(f1)
        with open(args.out, 'a') as outf:
            outf.write(f'{metrics}\n')

        with open(f'{args.out_num}_test_attr.json', 'w') as json_file:
            json.dump(test_attr_cache, json_file)
        serializable_keyword_pool = {key: list(value) if isinstance(value, set) else value
                                     for key, value in keyword_pool.items()}
        with open(f'{args.out_num}_keyword_pool.json', 'w') as json_file:
            json.dump(serializable_keyword_pool, json_file)

    print("DONE!")
