import re
import numpy as np
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import json
import utils
import generator
import apo_prompts


class PromptOptimizer(ABC):
    def __init__(self, args, evaluator_fn, scorer, max_threads=1, bf_eval=None):
        self.opt = args
        self.evaluator_fn = evaluator_fn
        self.scorer = scorer
        self.max_threads = max_threads
        self.bf_eval = bf_eval

    @abstractmethod
    def expand_candidates(self, prompts, task, gpt4, train_exs, attribute_cache):
        pass


class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """

    def _sample_error_str(self, texts, labels, preds, task, attributes=[], n=4):
        """ Sample n error strings from the given texts, labels, and preds"""
        # samples a few examples where the model's predictions do not match the true labels
        # then generates a formatted error string that summarizes these mismatches for easier analysis
        error_idxs = []
        for i, (l, p) in enumerate(zip(labels, preds)):
            if l != p:
                error_idxs.append(i)

        sample_idxs = random.sample(error_idxs, min(len(error_idxs), n))

        sample_texts = [texts[i] for i in sample_idxs]
        sample_labels = [labels[i] for i in sample_idxs]
        sample_preds = [preds[i] for i in sample_idxs]

        error_string = ''
        error_idx = 0
        if len(attributes) > 0:
            sample_attrs = [attributes[i] for i in sample_idxs]
            for i, (l, p, a) in enumerate(zip(sample_labels, sample_preds, sample_attrs)):
                error_string += f'## Image {error_idx + 1}\n'
                error_string += f'Text: The image shows the following features: \"{a.strip()}\"\nLabel: {task.stringify_prediction(l)}\nPrediction: {task.stringify_prediction(p)}\n\n'
                error_idx += 1
            return error_string.strip(), sample_texts
        else:
            for i, (t, l, p) in enumerate(zip(sample_texts, sample_labels, sample_preds)):
                error_string += f'## Example {error_idx + 1}\n'
                error_string += f'Text: \"{t.strip()}\"\nLabel: {task.stringify_prediction(l)}\nPrediction: {task.stringify_prediction(p)}\n\n'
                error_idx += 1
            return error_string.strip(), []

    def parse_tagged_text(self, text, start_tag, end_tag):
        """ Parse text that is tagged with start and end tags."""
        texts = []
        while True:
            start_index = text.find(start_tag)
            if start_index == -1:
                break
            end_index = text.find(end_tag, start_index)
            if end_index == -1:
                break
            start_index += len(start_tag)
            select = text[start_index:end_index].strip().strip('"').strip('`')
            if "` and `" not in select and select != "and" and select != " and ":
                texts.append(text[start_index:end_index].strip())
            text = text[end_index + len(end_tag):]
        return texts

    def _get_gradients(self, prompt, error_string, img_paths=[], num_feedbacks=5, n=1):
        """ Get "gradients" for a prompt based on the error string."""
        if self.opt['task'] == 'BRACS' and self.opt['class0'] == 'N':
            gradient_prompt = apo_prompts.BRACS_N_IC
        elif self.opt['task'] == 'BRACS' and self.opt['class0'] == 'DCIS':
            gradient_prompt = apo_prompts.BRACS_DCIS_IC
        elif self.opt['task'] == 'BACH' and self.opt['class0'] == 'N':
            gradient_prompt = apo_prompts.BACH_N_I
        elif self.opt['task'] == 'SICAPv2' and self.opt['class0'] == 'N':
            gradient_prompt = apo_prompts.SICAPv2_N_C
        elif 'multi' in self.opt['task']:
            gradient_prompt = apo_prompts.BRACS_multi
        else:
            raise Exception(f"Unsupported task: {self.opt['task_name']}")
        gradient_prompt += f"""
        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_string}

        Give a reasons why the prompt could have gotten these examples wrong.
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])
        if len(img_paths) > 0:
            if 'gemini' in self.opt['gradient_model']:
                res = utils.google_gemini(gradient_prompt, img_paths)
            else:
                res = utils.gpt4o(gradient_prompt, img_paths, n=n)
        else:
            if 'gemini' in self.opt['gradient_model']:
                res = utils.google_gemini(gradient_prompt)
            else:
                res = utils.gpt4o(gradient_prompt, n=n)

        with open(self.opt['out'], 'a') as outf:
            outf.write('feedbacks: ' + json.dumps(res) + '\n')
        return res

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, img_paths=[], n=1):
        """ Incorporate feedback gradient into a prompt."""
        if self.opt['task'] == 'BRACS' and self.opt['class0'] == 'N':
            transformation_prompt = apo_prompts.BRACS_N_IC
        elif self.opt['task'] == 'BRACS' and self.opt['class0'] == 'DCIS':
            transformation_prompt = apo_prompts.BRACS_DCIS_IC
        elif self.opt['task'] == 'BACH' and self.opt['class0'] == 'N':
            transformation_prompt = apo_prompts.BACH_N_I
        elif self.opt['task'] == 'SICAPv2' and self.opt['class0'] == 'N':
            transformation_prompt = apo_prompts.SICAPv2_N_C
        elif 'multi' in self.opt['task']:
            transformation_prompt = apo_prompts.BRACS_multi
        else:
            raise Exception(f"Unsupported task: {self.opt['task_name']}")
        transformation_prompt += f"""
        My current prompt is:
        "{prompt}"

        But this prompt gets the following examples wrong:
        {error_str}

        Based on these examples the problem with this prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} different improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])
        if len(img_paths) > 0:
            if 'gemini' in self.opt['gradient_model']:
                res = utils.google_gemini(transformation_prompt, img_paths)
            else:
                res = utils.gpt4o(transformation_prompt, img_paths, n=n)
        else:
            if 'gemini' in self.opt['gradient_model']:
                res = utils.google_gemini(transformation_prompt)
            else:
                res = utils.gpt4o(transformation_prompt, n=n)
        new_prompts = []
        for r in res:
            new_prompts += self.parse_tagged_text(r, "<START>", "<END>")
        with open(self.opt['out'], 'a') as outf:
            for p in new_prompts:
                outf.write('new prompts: ' + json.dumps(p) + '\n')
            if len(new_prompts) == 0:
                outf.write('new prompts: None. Dumping res:' + json.dumps(res) + '\n')
        return new_prompts

    def generate_synonyms(self, prompt_section, n=3):
        """ Generate synonyms for a prompt section."""
        rewriter_prompt = f"Generate one variation of the following instruction while keeping the semantic meaning.\n\nInput: {prompt_section}\n\nOutput: ## Variation:\n\n"
        if 'gemini' in self.opt['gradient_model']:
            new_instructions = []
            for i in range(n):
                new_instructions.append(utils.google_gemini(rewriter_prompt)[0])
        else:
            new_instructions = utils.gpt4o(rewriter_prompt, n=n)
        new_instructions = [re.sub(r"^## Variation:\s*", "", x).strip().strip('"').strip('`')
                            for x in new_instructions if x]
        for x in new_instructions:
            with open(self.opt['out'], 'a') as outf:
                outf.write('synonyms: ' + json.dumps(x) + '\n')
        return new_instructions

    def get_gradients(self, task_section, task, texts, labels, preds, attributes):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string, img_paths = self._sample_error_str(texts, labels, preds, task, attributes,
                                                             n=self.opt['errors_per_gradient'])
            gradients = self._get_gradients(task_section, error_string, img_paths, self.opt['gradients_per_error'], n=1)
            prompt_feedbacks += [(t, error_string, img_paths) for t in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts, task, gpt4, train_exs, pred_prompt=None, attribute_cache=None):
        """ Expand a list of prompts by generating gradient-based successors and
            synonyms for each section.
        """
        minibatch = random.sample(train_exs, k=min(len(train_exs), self.opt['minibatch_size']))

        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections['task'].strip()

            # evaluate prompt on minibatch
            _, texts, labels, preds, attributes = task.evaluate(gpt4, prompt, minibatch, pred_prompt=pred_prompt,
                                                                attribute_cache=attribute_cache)

            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                # [(gradient1, error1), (gradient2, error1), ...]
                gradients = self.get_gradients(task_section, task, texts, labels, preds, attributes)
                new_task_sections = []
                for feedback, error_string, img_paths in tqdm(gradients, desc='applying gradients'):
                    tmp = self.apply_gradient(task_section, error_string, feedback,
                                              self.opt['steps_per_gradient'], img_paths)
                    new_task_sections += tmp

            # generate synonyms
            mc_sampled_task_sections = []
            if self.opt['mc_samples_per_step'] > 0:
                for sect in tqdm(new_task_sections + [task_section], desc='mc samples'):
                    mc_sects = self.generate_synonyms(sect, n=self.opt['mc_samples_per_step'])
                    mc_sampled_task_sections += mc_sects

            # combine
            new_sections = new_task_sections + mc_sampled_task_sections
            new_sections = list(set(new_sections))  # dedup
            tmp_new_prompts = [prompt.replace(task_section, tmp.strip()) for tmp in new_sections
                               if tmp != task_section and prompt.replace(task_section, tmp) != prompt]
            if len(tmp_new_prompts) != len(new_sections):
                print(f'Invalid replacement with {len(tmp_new_prompts)} != {len(new_sections)}')
                with open(self.opt['out'], 'a') as outf:
                    outf.write(f'{len(tmp_new_prompts)}, {len(new_sections)}: ' + json.dumps(task_section) + '\n')

            # filter a little
            if len(new_sections) > self.opt['max_expansion_factor']:
                if self.opt['reject_on_errors']:
                    error_exs = []
                    for i, (t, l, p) in enumerate(zip(texts, labels, preds)):
                        if l != p:
                            error_exs.append({'text': t, 'label': l, 'img_path': t})
                    error_exs = random.sample(error_exs, min(len(error_exs), 16))
                    # speed up a little
                    tmp_new_prompts = random.sample(tmp_new_prompts,
                                                    min(len(tmp_new_prompts), self.opt['max_expansion_factor'] * 2))

                    if attribute_cache != None:
                        print('Generating temp attr cache:', len(tmp_new_prompts))
                        temp_attr_cache = {}
                        for temp_prompt in tmp_new_prompts:
                            temp_attr_cache[f'{temp_prompt}'] = {}
                            temp_attr_cache = generator.parallel_generate(generator.AttrGredictor(self.opt),
                                                                          temp_prompt, error_exs, temp_attr_cache, 16)
                        error_scores = self.bf_eval(tmp_new_prompts, error_exs, gpt4, self.scorer, pred_prompt,
                                                    temp_attr_cache, max_threads=self.max_threads)
                    else:
                        error_scores = self.bf_eval(tmp_new_prompts, error_exs, gpt4, self.scorer,
                                                    max_threads=self.max_threads)
                    tmp_new_prompts = [tmp_new_prompts[i]
                                       for i in np.argsort(error_scores)[-self.opt['max_expansion_factor']:]]
                else:
                    tmp_new_prompts = random.sample(tmp_new_prompts, k=self.opt['max_expansion_factor'])

            new_prompts += tmp_new_prompts

        new_prompts += prompts  # add originals
        new_prompts = list(set(new_prompts))  # dedup

        return new_prompts

    def score_candidates(self, prompts, gpt4, train_exs, pred_prompt=None, attribute_cache=None):
        """ Score a list of prompts."""
        if len(prompts) == 1:
            return [1.0]

        evals = self.evaluator_fn(prompts, train_exs, gpt4,
                                  scorer=self.scorer,
                                  pred_prompt=pred_prompt,
                                  attribute_cache=attribute_cache,
                                  rounds=self.opt['eval_rounds'],
                                  num_prompts_per_round=self.opt['eval_prompts_per_round'],
                                  samples_per_eval=self.opt['samples_per_eval'],
                                  max_threads=self.max_threads)
        return evals
