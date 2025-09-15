import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import evaluators
from tqdm import tqdm
import time
import datetime
import json
import random
import argparse
import utils
import scorers
import tasks
import predictors
import optimizers
import generator

random.seed(42)


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


def get_evaluator(evaluator):
    if evaluator == 'bf':
        return evaluators.BruteForceEvaluator
    elif evaluator in {'ucb', 'ucb-e'}:
        return evaluators.UCBBanditEvaluator
    elif evaluator in {'sr', 's-sr'}:
        return evaluators.SuccessiveRejectsEvaluator
    elif evaluator == 'sh':
        return evaluators.SuccessiveHalvingEvaluator
    else:
        raise Exception(f'Unsupported evaluator: {evaluator}')


def get_scorer(scorer):
    if scorer == '01':
        return scorers.Cached01Scorer
    else:
        raise Exception(f'Unsupported scorer: {scorer}')


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
    parser.add_argument('--get_even_exs', action='store_true', default=False)
    parser.add_argument('--n_train', default=20, type=int, help='# instances per class')
    parser.add_argument('--n_val', default=15, type=int)
    parser.add_argument('--n_test', default=10, type=int)

    parser.add_argument('--minibatch_size', default=24, type=int)
    parser.add_argument('--n_gradients', default=3, type=int, help='# generated gradients per prompt')
    parser.add_argument('--errors_per_gradient', default=4, type=int,
                        help='# error examples used to generate one gradient')
    parser.add_argument('--gradients_per_error', default=1, type=int, help='# gradient reasons per error')
    parser.add_argument('--steps_per_gradient', default=1, type=int, help='# new prompts per gradient reason')
    parser.add_argument('--mc_samples_per_step', default=1, type=int, help='# synonyms')
    parser.add_argument('--max_expansion_factor', default=5, type=int, help='maximum # prompts after expansion')

    parser.add_argument('--evaluator', default="ucb", type=str)
    parser.add_argument('--scorer', default="01", type=str)
    parser.add_argument('--eval_rounds', default=8, type=int)
    parser.add_argument('--eval_prompts_per_round', default=8, type=int)
    # calculated by s-sr and sr
    parser.add_argument('--samples_per_eval', default=32, type=int)
    parser.add_argument('--c', default=1.0, type=float, help='exploration param for UCB. higher = more exploration')
    parser.add_argument('--knn_k', default=2, type=int)
    parser.add_argument('--knn_t', default=0.993, type=float)
    parser.add_argument('--diverse_init', action='store_true', default=False)
    parser.add_argument('--diverse_exp', default=2, type=int)
    parser.add_argument('--reject_on_errors', action='store_true', default=False)
    parser.add_argument('--val_score', action='store_true', default=False)
    parser.add_argument('--seed', default=42, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if 'multi' in args.task:
        args.out = f'results/{args.out_num}_{args.task}_{args.model}_test_out.txt'
    else:
        args.out = f'results/{args.out_num}_{args.task}_{args.model}_{args.class0}_{args.class1}_test_out.txt'

    configs = vars(args)
    configs['eval_budget'] = configs['samples_per_eval'] * configs['eval_rounds'] * configs['eval_prompts_per_round']
    if "gemini" in args.model:
        utils.clear_gemini_img_files(True)

    task = get_task_class(args)
    scorer = get_scorer(args.scorer)()
    evaluator = get_evaluator(args.evaluator)(configs)
    bf_eval = get_evaluator('bf')(configs)
    gpt4 = get_predictor(configs)
    gpt_generator = generator.AttrGredictor(configs)
    optimizer = optimizers.ProTeGi(configs, evaluator, scorer, args.max_threads, bf_eval)

    if os.path.exists(args.out):
        os.remove(args.out)
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')

    if 'multi' in args.task:
        train_exs = task.get_even_exs('train', args.n_train, seed=args.seed)
        val_exs = task.get_even_exs('val', args.n_val, seed=args.seed)
        test_exs = task.get_even_exs('test', args.n_val, seed=args.seed)
    else:
        if args.get_even_exs:
            train_exs = task.get_even_exs('train', args.n_train, seed=args.seed)
            # test_exs = task.get_even_exs('test', args.n_test, seed=args.seed)
            with open(f'../file_names/test_exs_{args.class0}_{args.class1}.json', 'r') as json_file:
                test_exs = json.load(json_file)
            val_exs = task.get_even_exs('val', args.n_val, seed=args.seed)
        else:
            with open(f'../file_names/train_exs_{args.class0}_{args.class1}.json', 'r') as json_file:
                train_exs = json.load(json_file)
            with open(f'../file_names/test_exs_{args.class0}_{args.class1}.json', 'r') as json_file:
                test_exs = json.load(json_file)
            val_exs = task.get_examples('val')

    if args.diverse_init:
        with open(f'../diversity/results/{args.diverse_exp}_analysis/{args.diverse_exp}_test_attr.json', 'r') as json_file:
            attr_all = json.load(json_file)
        prompt_keys = list(attr_all.keys())
        candidates = prompt_keys[-4:]
    else:
        candidates = [open(f'../prompts/{args.task}_generate.md').read()]
    if 'multi' in args.task:
        pred_prompt = open(f'../prompts/{args.task}.md').read()
    else:
        pred_prompt = open(f'../prompts/{args.task}_{args.class0}_{args.class1}.md').read()
    with open(args.out, 'a') as outf:
        outf.write(f'pred_prompt-------------------------\n')
        outf.write(f'{pred_prompt}\n\n')

    attribute_cache, test_attr_cache = {}, {}
    for prompt in candidates:
        attribute_cache[f'{prompt}'] = {}
        attribute_cache = generator.parallel_generate(gpt_generator, prompt, train_exs, attribute_cache, args.max_threads)
        test_attr_cache[f'{prompt}'] = {}
        test_attr_cache = generator.parallel_generate(gpt_generator, prompt, test_exs, test_attr_cache, args.max_threads)
    val_attr_cache = {}
    if args.val_score:
        for prompt in candidates:
            val_attr_cache[f'{prompt}'] = {}
            val_attr_cache = generator.parallel_generate(gpt_generator, prompt, val_exs, val_attr_cache, args.max_threads)

    for round in tqdm(range(configs['rounds'] + 1)):
        print("STARTING ROUND ", round)
        with open(args.out, 'a') as outf:
            outf.write(f"======== ROUND {round}\n")
        start = time.time()

        if round > 0:
            candidates = optimizer.expand_candidates(candidates, task, gpt4, train_exs, pred_prompt, attribute_cache)
            for prompt in candidates:
                # if f'{prompt}' not in attribute_cache:
                if args.val_score:
                    val_attr_cache[f'{prompt}'] = {}
                    val_attr_cache = generator.parallel_generate(gpt_generator, prompt, val_exs, val_attr_cache, args.max_threads)
                else:
                    attribute_cache[f'{prompt}'] = {}
                    attribute_cache = generator.parallel_generate(gpt_generator, prompt, train_exs, attribute_cache, args.max_threads)

        if args.val_score:
            scores = optimizer.score_candidates(candidates, gpt4, val_exs, pred_prompt, val_attr_cache)
        else:
            scores = optimizer.score_candidates(candidates, gpt4, train_exs, pred_prompt, attribute_cache)
        [scores, candidates] = list(zip(*sorted(list(zip(scores, candidates)), reverse=True)))

        candidates = candidates[:configs['beam_size']]
        scores = scores[:configs['beam_size']]

        with open(args.out, 'a') as outf:
            outf.write(f'{time.time() - start}\n')
            # outf.write(f'{candidates}\n')
            for c in candidates:
                outf.write(json.dumps(c) + '\n')
            outf.write(f'{scores}\n')

        metrics = []
        for prompt in candidates:
            if args.val_score:
                attribute_cache[f'{prompt}'] = {}
                attribute_cache = generator.parallel_generate(gpt_generator, prompt, train_exs, attribute_cache, args.max_threads)
            if f'{prompt}' not in test_attr_cache:
                test_attr_cache[f'{prompt}'] = {}
                test_attr_cache = generator.parallel_generate(gpt_generator, prompt, test_exs, test_attr_cache, args.max_threads)

        for candidate, score in zip(candidates, scores):
            f1, texts, labels, preds, attr = task.evaluate(gpt4, candidate, test_exs, pred_prompt=pred_prompt, attribute_cache=test_attr_cache)
            metrics.append(f1)
        with open(args.out, 'a') as outf:
            outf.write(f'{metrics}\n')

        with open(f'{args.out_num}_train_attr.json', 'w') as json_file:
            json.dump(attribute_cache, json_file)
        with open(f'{args.out_num}_test_attr.json', 'w') as json_file:
            json.dump(test_attr_cache, json_file)
        if args.val_score:
            with open(f'{args.out_num}_val_attr.json', 'w') as json_file:
                json.dump(val_attr_cache, json_file)

    print("DONE!")
