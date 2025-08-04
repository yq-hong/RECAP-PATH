import json
import argparse
import spacy
from nltk.stem import WordNetLemmatizer
import string
import re


def clean_text(text):
    text = text.replace('-', '.')  # Replace hyphens with periods
    text = re.sub(r'[^\w\s,.]', ' ', text)  # Replace punctuation with spaces
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = re.sub(r'\s+', ' ', text)  # Normalize multiple spaces to a single space
    text = text.strip().lower()  # Convert to lowercase and strip leading/trailing spaces
    if text.startswith('task '):  # Remove 'task' at the start
        text = text[5:]  # Remove the first 5 characters ("task ")
    return text


def get_text_keywords(text):
    cleaned_text = clean_text(text)
    cleaned_text = cleaned_text.translate(str.maketrans('', '', string.punctuation))
    lemmatizer = WordNetLemmatizer()

    nlp = spacy.load("en_core_sci_sm")
    doc = nlp(cleaned_text)
    # keywords = set(ent.text for ent in doc.ents)
    keywords = set(lemmatizer.lemmatize(ent.text) for ent in doc.ents)
    return keywords


def get_diversity_score(prompt_keywords):
    all_keywords = set.union(*prompt_keywords.values())

    diversity_ratio = {}
    max_diversity = max(len(value) for value in prompt_keywords.values())
    for prompt, keywords in prompt_keywords.items():
        total = len(keywords)
        diversity = total / max_diversity if max_diversity > 0 else 0
        diversity_ratio[prompt] = diversity

    prompt_keys = list(prompt_keywords.keys())
    unique_keywords = {
        prompt: keywords - set.union(set(), *(prompt_keywords[p] for p in prompt_keys if p != prompt))
        for prompt, keywords in prompt_keywords.items()
    }

    uniqueness_ratios = {}
    max_uniqueness = max(len(value) for value in unique_keywords.values())
    for prompt, keywords in prompt_keywords.items():
        unique = unique_keywords[prompt]
        uniqueness = len(unique) / max_uniqueness if max_uniqueness > 0 else 0
        uniqueness_ratios[prompt] = uniqueness

    diversity_score = {}
    for prompt, keywords in prompt_keywords.items():
        diversity_score[prompt] = diversity_ratio[prompt] + uniqueness_ratios[prompt]

    return diversity_score, diversity_ratio, uniqueness_ratios, unique_keywords


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='BRACS')
    parser.add_argument('--result_folder', default='diversity')
    parser.add_argument('--exp', default=1, type=int)

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()

    with open(f'../{args.result_folder}/results/{args.exp}_analysis/{args.exp}_test_attr.json', 'r') as json_file:
        attr_all = json.load(json_file)

    prompt_keys = list(attr_all.keys())
    prompt_keywords = {}

    for count, prompt in enumerate(prompt_keys):
        prompt_keywords[prompt] = get_text_keywords(prompt)

    diversity_score, diversity_ratio, uniqueness_ratios, unique_keywords = get_diversity_score(prompt_keywords)

    output_file = f'keywords/{args.result_folder}_{args.exp}_processed_prompts.txt'
    with open(output_file, 'w') as f:
        f.write("Prompt Analysis Results\n\n")
        f.write(f"Diversity\tUniqueness\tDiversity Ratio\tUniqueness Ratio\tScore\t\n")
        for count, prompt in enumerate(prompt_keys):
            cleaned_prompt = clean_text(prompt)
            keywords = prompt_keywords[prompt]
            unique = unique_keywords[prompt]
            uniqueness = uniqueness_ratios[prompt]
            diversity = diversity_ratio[prompt]
            score = uniqueness + diversity
            # f.write(f"Prompt {count}:\n")
            # f.write(f"Original Prompt: {prompt}\n\n")
            # f.write(f"Cleaned Prompt: {cleaned_prompt}\n\n")
            # f.write(f"Keywords: {', '.join(keywords)}\n\n")
            # f.write(f"Unique Keywords: {', '.join(unique)}\n\n")
            # f.write(f"Diversity\tUniqueness\tDiversity Ratio\tUniqueness Ratio\tScore\t\n")
            f.write(f"{len(keywords)}\t{len(unique)}\t{diversity}\t{uniqueness}\t{score}\n")
            # f.write("\n--------------\n\n")

    print(f"Results written to {output_file}")
