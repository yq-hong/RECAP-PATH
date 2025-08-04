from abc import ABC, abstractmethod
from liquid import Template
import utils


class GPT4Predictor(ABC):
    def __init__(self, opt):
        self.opt = opt

    @abstractmethod
    def inference(self, ex, prompt):
        pass


class BinaryPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, ex, prompt):
        prompt = Template(prompt).render(text=ex['text'])
        response = utils.gpt4o(prompt, max_tokens=4, n=1, temperature=self.opt['temperature'])[0]
        pred = 1 if response.strip().upper().startswith('YES') else 0
        return pred


class TwoClassPredictor(GPT4Predictor):
    categories = ['No', 'Yes']

    def inference(self, prompt, img_paths=None, attr=None):
        prompt = Template(prompt).render(text=attr)

        if 'gemini' in self.opt['model']:
            response = utils.google_gemini(prompt, img_paths, max_tokens=12, temperature=self.opt['temperature'])[0]
        else:
            response = utils.gpt4o(prompt, img_paths, max_tokens=12, n=1, temperature=self.opt['temperature'])[0]

        if response is None:
            print("No response received from the model.")
            return None

        if 'A.' in response or '**A' in response or 'A\n' in response or 'A \n' in response or '(A)' in response or ': A' in response or response == 'A':
        # if 'No' in response:
            pred = 0
        elif 'B.' in response or '**B' in response or 'B\n' in response or 'B \n' in response or '(B)' in response or ': B' in response or response == 'B':
        # elif 'Yes' in response:
            pred = 1
        else:
            print(f"No valid response. {response}")
            return None
        # pred = 1 if response.strip().upper().startswith('YES') else 0

        return pred


class MultiClassPredictor(GPT4Predictor):

    def inference(self, prompt, img_paths=None, attr=None):
        prompt = Template(prompt).render(text=attr)

        if 'gemini' in self.opt['model']:
            response = utils.google_gemini(prompt, img_paths, max_tokens=6, temperature=self.opt['temperature'])[0]
        else:
            response = utils.gpt4o(prompt, img_paths, max_tokens=6, n=1, temperature=self.opt['temperature'])[0]
        if response is None:
            print("No response received from the model.")
            return None

        if 'A.' in response or '**A' in response or 'A \n' in response or 'A\n' in response or '(A)' in response or ': A' in response or response == 'A':
            pred = 0
        elif 'B.' in response or '**B' in response or 'B \n' in response or 'B\n' in response or '(B)' in response or ': B' in response or response == 'B':
            pred = 1
        elif 'C.' in response or '**C' in response or 'C \n' in response or 'C\n' in response or '(C)' in response or ': C' in response or response == 'C':
            pred = 2
        elif 'D.' in response or '**D' in response or 'D \n' in response or 'D\n' in response or '(D)' in response or ': D' in response or response == 'D':
            pred = 3
        elif 'E.' in response or '**E' in response or 'E \n' in response or 'E\n' in response or '(E)' in response or ': E' in response or response == 'E':
            pred = 4
        elif 'F.' in response or '**F' in response or 'F \n' in response or 'F\n' in response or '(F)' in response or ': F' in response or response == 'F':
            pred = 5
        elif 'G.' in response or '**G' in response or 'G \n' in response or 'G\n' in response or '(G)' in response or ': G' in response or response == 'G':
            pred = 6
        else:
            print(f"No valid response. {response}")
            return None

        return pred
