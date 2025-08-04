import os
import random
import requests
import json
import concurrent.futures
from abc import ABC, abstractmethod
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, classification_report
import generator

random.seed(42)


class DataProcessor(ABC):
    def __init__(self, data_dir, max_threads=1):
        self.data_dir = data_dir
        self.max_threads = max_threads

    @abstractmethod
    def evaluate(self, predictor, prompt, test_exs):
        pass

    @abstractmethod
    def stringify_prediction(self, pred):
        pass


def process_example(ex, predictor, pred_prompt, attr=None):
    img_path = ex['img_path']
    pred = predictor.inference(pred_prompt, [img_path], attr=attr)
    return ex, pred, attr


class ClassificationTask(DataProcessor):

    def run_evaluate(self, predictor, prompt, exs, pred_prompt=None, attribute_cache=None):
        labels = []
        preds = []
        texts = []
        attributes = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_threads) as executor:
            if attribute_cache != None:
                futures = [executor.submit(process_example, ex, predictor, pred_prompt, attribute_cache[f'{prompt}'][f'{ex}']) for ex in exs]
            else:
                futures = [executor.submit(process_example, ex, predictor, prompt, None) for ex in exs]
            for i, future in tqdm(enumerate(concurrent.futures.as_completed(futures)), total=len(futures), desc='running prediction on examples'):
                ex, pred, attr = future.result()
                if pred != None:
                    if attribute_cache != None:
                        texts.append(ex['img_path'])
                        attributes.append(attr)
                    else:
                        texts.append(ex['text'])
                    labels.append(ex['label'])
                    preds.append(pred)

        # accuracy = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='micro')
        return f1, texts, labels, preds, attributes

    def evaluate(self, predictor, prompt, test_exs, pred_prompt=None, attribute_cache=None):
        while True:
            try:
                f1, texts, labels, preds, attributes = self.run_evaluate(predictor, prompt, test_exs, pred_prompt, attribute_cache)
                break
            except (concurrent.futures.process.BrokenProcessPool, requests.exceptions.SSLError):
                pass
        return f1, texts, labels, preds, attributes


class BinaryClassificationTask(ClassificationTask):
    categories = ['No', 'Yes']

    def stringify_prediction(self, pred):
        return BinaryClassificationTask.categories[pred]


class BRACSBinaryTask(BinaryClassificationTask):
    def __init__(self, data_dir, max_threads=1, class0='N', class1='IC'):
        super().__init__(data_dir, max_threads)
        self.class0 = class0
        self.class1 = class1
        self.class2num = {'N': 0, 'PB': 1, 'UDH': 2, 'FEA': 3, 'ADH': 4, 'DCIS': 5, 'IC': 6}

    def stringify_prediction(self, pred):
        class2name = {'N': 'Normal', 'PB': 'Pathology Benign', 'UDH': 'Usual Ductal Hyperplasia',
                      'FEA': 'Flat Epithelial Atypia', 'ADH': 'Atypical Ductal Hyperplasia',
                      'DCIS': 'Ductal Carcinoma In Situ', 'IC': 'Invasive Carcinoma'}
        num2classname = [class2name[self.class0], class2name[self.class1]]
        return num2classname[pred]

    def get_examples(self, mode):
        meta_path0 = f'{self.data_dir}/{mode}_{self.class2num[self.class0]}_{self.class0}.txt'
        meta_path1 = f'{self.data_dir}/{mode}_{self.class2num[self.class1]}_{self.class1}.txt'

        exs = []
        count = 0
        with open(meta_path0, 'r') as file:
            for line in file:
                exs.append({'id': f'{count}', 'label': 0, 'img_path': os.path.join('/home/directory', line.strip())})
                count += 1
        with open(meta_path1, 'r') as file:
            for line in file:
                exs.append({'id': f'{count}', 'label': 1, 'img_path': os.path.join('/home/directory', line.strip())})
                count += 1

        return exs

    def get_few_shot_examples(self, train_exs, n_shots=1, seed=42):
        random.seed(seed)
        exs = [[], []]
        idxs = [[], []]

        for i in range(len(train_exs)):
            if train_exs[i]['label'] == 0:
                idxs[0].append(i)
            else:
                idxs[1].append(i)

        select0 = random.sample(idxs[0], n_shots)
        select1 = random.sample(idxs[1], n_shots)
        for idx in select0:
            exs[0].append(train_exs[idx])
        for idx in select1:
            exs[1].append(train_exs[idx])

        return exs

    def get_even_exs(self, mode='train', n_exs=100, seed=42):
        random.seed(seed)
        meta_path0 = f'{self.data_dir}/{mode}_{self.class2num[self.class0]}_{self.class0}.txt'
        meta_path1 = f'{self.data_dir}/{mode}_{self.class2num[self.class1]}_{self.class1}.txt'

        exs_0, exs_1 = [], []
        count0 = 0
        with open(meta_path0, 'r') as file:
            for line in file:
                exs_0.append({'id': f'{count0}', 'label': 0, 'img_path': os.path.join('/home/directory', line.strip())})
                count0 += 1
        exs_0 = random.sample(exs_0, min(len(exs_0), n_exs))
        count1 = 0
        with open(meta_path1, 'r') as file:
            for line in file:
                exs_1.append({'id': f'{count0 + count1}', 'label': 1, 'img_path': os.path.join('/home/directory', line.strip())})
                count1 += 1
        exs_1 = random.sample(exs_1, min(len(exs_1), n_exs))

        return exs_0 + exs_1

    def get_attr(self, args, prompt, exs, gpt_generator=None, generate=False):
        if generate:
            attribute_cache = {}
            attribute_cache[f'{prompt}'] = {}
            attribute_cache = generator.parallel_generate(gpt_generator, prompt, exs, attribute_cache, 16)
            attrs = {}
            for ex in exs:
                attrs[f"{ex['id']}"] = attribute_cache[f'{prompt}'][f'{ex}']
        else:
            with open(f'{args.result_folder}/results/{args.exp}_analysis/{args.exp}_{args.mode}_attr.json', 'r') as json_file:
                attr = json.load(json_file)
            attrs = {}
            for ex in exs:
                attrs[f"{ex['id']}"] = attr[f'{prompt}'][f'{ex}']
        return attrs


class BRACSMultiTask(BinaryClassificationTask):
    def stringify_prediction(self, pred):
        num2classname = {0: 'Normal', 1: 'Malignant (DCIS)', 2: 'Malignant (IC)'}
        return num2classname[pred]

    def get_examples(self, mode):
        file_paths = [f'{self.data_dir}/{mode}_0_N.txt',
                      # f'{self.data_dir}/{mode}_1_PB.txt', f'{self.data_dir}/{mode}_2_UDH.txt',
                      # f'{self.data_dir}/{mode}_3_FEA.txt', f'{self.data_dir}/{mode}_4_ADH.txt',
                      f'{self.data_dir}/{mode}_5_DCIS.txt', f'{self.data_dir}/{mode}_6_IC.txt']

        exs = []
        count = 0
        for i in range(len(file_paths)):
            with open(file_paths[i], 'r') as file:
                for line in file:
                    exs.append({'id': f'{count}', 'label': i, 'img_path': os.path.join('/home/directory', line.strip())})
                    count += 1

        return exs

    def get_few_shot_examples(self, train_exs, n_shots=1, seed=42):
        random.seed(seed)
        exs = [[], []]
        idxs = [[], []]

        for i in range(len(train_exs)):
            if train_exs[i]['label'] == 0:
                idxs[0].append(i)
            else:
                idxs[1].append(i)

        select0 = random.sample(idxs[0], n_shots)
        select1 = random.sample(idxs[1], n_shots)
        for idx in select0:
            exs[0].append(train_exs[idx])
        for idx in select1:
            exs[1].append(train_exs[idx])

        return exs

    def get_even_exs(self, mode='train', n_exs=100, seed=42):
        random.seed(seed)
        file_paths = [f'{self.data_dir}/{mode}_0_N.txt',
                      # f'{self.data_dir}/{mode}_1_PB.txt', f'{self.data_dir}/{mode}_2_UDH.txt',
                      # f'{self.data_dir}/{mode}_3_FEA.txt', f'{self.data_dir}/{mode}_4_ADH.txt',
                      f'{self.data_dir}/{mode}_5_DCIS.txt', f'{self.data_dir}/{mode}_6_IC.txt']

        exses = [[] for _ in range(len(file_paths))]
        count = 0
        counts = [0 for _ in range(len(file_paths))]
        for i in range(len(file_paths)):
            with open(file_paths[i], 'r') as file:
                for line in file:
                    exses[i].append({'id': f'{count}', 'label': i, 'img_path': os.path.join('/home/directory', line.strip())})
                    counts[i] += 1
                    count += 1
            exses[i] = random.sample(exses[i], min(len(exses[i]), n_exs))

        exs = []
        for ex_list in exses:
            exs += ex_list

        return exs

    def get_attr(self, args, prompt, exs, gpt_generator=None, generate=False):
        if generate:
            attribute_cache = {}
            attribute_cache[f'{prompt}'] = {}
            attribute_cache = generator.parallel_generate(gpt_generator, prompt, exs, attribute_cache, 16)
            attrs = {}
            for ex in exs:
                attrs[f"{ex['id']}"] = attribute_cache[f'{prompt}'][f'{ex}']
        else:
            with open(f'{args.result_folder}/results/{args.exp}_analysis/{args.exp}_{args.mode}_attr.json', 'r') as json_file:
                attr = json.load(json_file)
            attrs = {}
            for ex in exs:
                attrs[f"{ex['id']}"] = attr[f'{prompt}'][f'{ex}']
        return attrs
