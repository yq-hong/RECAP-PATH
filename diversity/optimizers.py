import sys
import os.path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
import re
from tqdm import tqdm
import random
from abc import ABC, abstractmethod
import json
import utils
import prompt_optimization.apo_prompts as apo_prompts

random.seed(42)


class PromptOptimizer(ABC):
    def __init__(self, args, max_threads=1):
        self.opt = args
        self.max_threads = max_threads

    @abstractmethod
    def expand_candidates(self, prompts, prompt_pool):
        pass


class ProTeGi(PromptOptimizer):
    """ ProTeGi: Prompt Optimization with Textual Gradients
    """

    def _sample_error_str(self, prompt_pool):
        sorted_prompt_pool = dict(sorted(prompt_pool.items(), key=lambda item: item[1], reverse=True))
        N = len(sorted_prompt_pool)
        if N > 2:
            select_idx = random.sample(range(1, N - 1), min(3, N - 2))
        else:
            select_idx = []

        example_string = ''
        idx, count = 0, 0
        for prompt in sorted_prompt_pool:
            if idx == 0 or idx == (N - 1) or idx in select_idx:
                section = utils.parse_sectioned_prompt(prompt)['task'].strip()
                example_string += f'## Example prompt {count + 1}:\n{section}\nDiversity score: {sorted_prompt_pool[prompt]}\n\n'
                count += 1
            idx += 1
        return example_string.strip()

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

    def _get_gradients(self, prompt, error_string, num_feedbacks=5, n=1):
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
        Here are some example prompts with their corresponding diversity scores. The higher the score, the more diverse and unique the prompt is.
        
        {error_string}
        
        My current prompt is:
        "{prompt}"

        Give a specific reason why my current prompt is not diverse enough. 
        Focus specifically on terminology. Give either existing terminology in the prompt that should be removed, or provide new terminology that is useful for classification.
        Do not correct for big picture issues like "the prompt is too vague" or "the prompt is too specific".
        """
        gradient_prompt = '\n'.join([line.lstrip() for line in gradient_prompt.split('\n')])

        if 'gemini' in self.opt['gradient_model']:
            res = utils.google_gemini(gradient_prompt)
        else:
            res = utils.gpt4o(gradient_prompt, n=n)

        with open(self.opt['out'], 'a') as outf:
            outf.write('feedbacks: ' + json.dumps(res) + '\n')
        return res

    def apply_gradient(self, prompt, error_str, feedback_str, steps_per_gradient, n=1):
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
        Here are some example prompts with their corresponding diversity scores. The higher the score, the more diverse and unique the prompt is.
        
        {error_str}
        
        My current prompt is:
        "{prompt}"

        Based on these examples the problem with my current prompt is that {feedback_str}

        Based on the above information, I wrote {steps_per_gradient} improved prompts.
        Each prompt is wrapped with <START> and <END>.

        The {steps_per_gradient} new prompts are:
        """
        transformation_prompt = '\n'.join([line.lstrip() for line in transformation_prompt.split('\n')])

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

    def get_gradients(self, task_section, prompt_pool):
        """ Get "gradients" for a prompt based on sampled error strings."""
        prompt_feedbacks = []
        for _ in tqdm(range(self.opt['n_gradients']), total=self.opt['n_gradients'], desc='gradients..'):
            error_string = self._sample_error_str(prompt_pool)
            gradients = self._get_gradients(task_section, error_string, self.opt['gradients_per_error'], n=1)
            prompt_feedbacks += [(t, error_string) for t in gradients]
        return prompt_feedbacks

    def expand_candidates(self, prompts, prompt_pool):
        """ Expand a list of prompts by generating gradient-based successors and
            synonyms for each section.
        """
        new_prompts = []
        for prompt in tqdm(prompts, desc=f'expanding {len(prompts)} prompts'):
            sections = utils.parse_sectioned_prompt(prompt)
            task_section = sections['task'].strip()

            # get gradients
            new_task_sections = []
            if self.opt['n_gradients'] > 0:
                # [(gradient1, error1), (gradient2, error1), ...]
                gradients = self.get_gradients(task_section, prompt_pool)
                new_task_sections = []
                for feedback, error_string in tqdm(gradients, desc='applying gradients'):
                    tmp = self.apply_gradient(task_section, error_string, feedback, self.opt['steps_per_gradient'])
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

            new_prompts += tmp_new_prompts

        new_prompts = list(set(new_prompts))  # dedup

        return new_prompts
