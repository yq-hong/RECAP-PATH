import os.path
import json
import argparse
import datetime
from tqdm import tqdm
import concurrent.futures
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import predictors
import tasks
import generator


def get_task_class(args):
    if args.task == 'BRACS':
        return tasks.BRACSBinaryTask(args.data_dir, args.max_threads, args.class0, args.class1)
    elif args.task == "BACH":
        return tasks.BACHBinaryTask(
            args.data_dir, args.max_threads, args.class0, args.class1
        )
    elif args.task == "SICAPv2":
        return tasks.SICAPv2BinaryTask(
            args.data_dir, args.max_threads, args.class0, args.class1
        )
    elif args.task == "Gleason":
        return tasks.GleasonBinaryTask(
            args.data_dir, args.max_threads, args.class0, args.class1
        )
    elif args.task == 'BRACS_multi':
        return tasks.BRACSMultiTask(args.data_dir, args.max_threads)
    else:
        raise Exception(f'Unsupported task: {args.task}')


def get_predictor(configs):
    if (
        configs['task'] == 'BRACS' 
        or configs["task"] == "BACH"
        or configs["task"] == "SICAPv2"
        or configs["task"] == "Gleason"
    ):
        return predictors.TwoClassPredictor(configs)
    elif configs['task'] == 'BRACS_multi':
        return predictors.MultiClassPredictor(configs)
    else:
        raise Exception(f"Unsupported task: {configs['task']}")


def process_example(ex, predictor, prompt, attr):
    img_path = ex['img_path']
    pred = predictor.inference(prompt, [img_path], attr=attr)
    return ex, pred


def run_evaluate(predictor, prompt, exs, attributes_dict):
    ids = []
    labels = []
    preds = []
    img_paths = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [executor.submit(process_example, ex, predictor, prompt, attributes_dict[f'{ex["id"]}']) for ex in exs]
        for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running prediction on examples'):
            ex, pred = future.result()
            if pred != None:
                img_paths.append(ex['img_path'])
                labels.append(ex['label'])
                preds.append(pred)
                ids.append(ex['id'])
            else:
                print(f"No prediction for {ex['id']}\t{ex['img_path']}")
                with open(args.out, 'a') as outf:
                    outf.write(f"No prediction for {ex['id']}\t{ex['img_path']}\n")

    correct_count = sum(1 for a, b in zip(labels, preds) if a == b)
    accuracy = correct_count / len(exs)
    # accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    conf_matrix = confusion_matrix(labels, preds)
    return f1, accuracy, conf_matrix, img_paths, labels, preds, ids


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='BRACS', choices=['BRACS', 'BRACS_multi'])
    parser.add_argument('--model', default='gemini')
    parser.add_argument('--out_num', default='0')
    parser.add_argument('--data_dir', default='file_names')
    parser.add_argument('--result_folder', default='diversity')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--class0', default='N')
    parser.add_argument('--class1', default='IC')
    parser.add_argument('--max_threads', default=16, type=int)
    parser.add_argument('--max_token', default=1024, type=int)
    parser.add_argument('--temperature', default=0.0, type=float)
    parser.add_argument('--generate', action='store_true', default=False)
    parser.add_argument('--all_exs', action='store_true', default=False)
    parser.add_argument('--n_test', default=10, type=int)
    parser.add_argument('--prompt_idx', default=0, type=int)
    parser.add_argument('--exp', default=1, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if 'multi' in args.task:
        args.out = f'{args.result_folder}/results/{args.exp}_analysis/evaluate/{args.exp}_{args.task}_{args.model}_prompt{args.prompt_idx}_{args.out_num}.txt'
        prompt_path = f'prompts/{args.task}.md'
    else:
        args.out = f'{args.result_folder}/results/{args.exp}_analysis/evaluate/{args.exp}_{args.task}_{args.model}_{args.class0}_{args.class1}_prompt{args.prompt_idx}_{args.out_num}.txt'
        prompt_path = f'prompts/{args.task}_{args.class0}_{args.class1}.md'
    if os.path.exists(args.out):
        os.remove(args.out)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    configs = vars(args)
    with open(args.out, 'a') as outf:
        outf.write(f'{str(datetime.datetime.now())}\n')
        outf.write(json.dumps(configs) + '\n')

    task = get_task_class(args)
    gpt4 = get_predictor(configs)
    gpt_generator = generator.AttrGredictor(configs)

    prompt = open(prompt_path).read()
    with open(args.out, 'a') as outf:
        outf.write(f'prompt-------------------------\n')
        outf.write(f'{prompt}\n\n')

    if 'multi' in args.task:
        exs = task.get_even_exs(args.mode, args.n_test)
    else:
        if args.all_exs:
            exs = task.get_examples(args.mode)
        else:
            with open(f'file_names/{args.mode}_exs_{args.class0}_{args.class1}.json', 'r') as json_file:
                exs = json.load(json_file)

    with open(f'{args.result_folder}/results/{args.exp}_analysis/{args.exp}_test_attr.json', 'r') as json_file:
        attr_all = json.load(json_file)
    prompt_keys = list(attr_all.keys())
    generate_prompt = prompt_keys[args.prompt_idx]
    with open(args.out, 'a') as outf:
        outf.write(f'generate_prompt-------------------------\n')
        outf.write(f'{generate_prompt}\n\n')

    attrs = task.get_attr(args, generate_prompt, exs, gpt_generator, args.generate)

    f1, acc, conf_matrix, texts, labels, preds, ids = run_evaluate(gpt4, prompt, exs, attrs)

    with open(args.out, 'a') as outf:
        outf.write(f'\nAccuracy: {acc}\tF1: {f1}\n')
        outf.write(f"Confusion Matrix:\n{conf_matrix}\n\n")
        outf.write('\nID\tPred\tTruth\tImage Path\n')
        for i in range(len(labels)):
            if labels[i] != preds[i]:
                outf.write(f'{ids[i]}\t{preds[i]}\t{labels[i]}\t{texts[i]}\n')

    with open(args.out, 'a') as outf:
        for ex in exs:
            idx = ex['id']
            outf.write(f"\n{ex['id']}\t{ex['label']}\t{ex['img_path']}-----------------------\n")
            outf.write(f"{attrs[f'{idx}']}\n\n")

    print('DONE!')
