"""
Generates adapted few-shot for singe-entity generative extraction
"""

import json
import random

with open("data/hdi_extractive_entity_recognition.json", 'r') as fp:
    data = json.load(fp)['dataset']

with open('data/labels.txt', 'r') as fp:
    labels = [elt.strip() for elt in fp.readlines()]

for label in labels:
    clean_label = label.capitalize()
    possible_shots = []
    for line in data:
        if clean_label in line[2].keys():
            possible_shots.append(line)
    random.shuffle(possible_shots)
    json_structure = {'shots': []}
    for i in range(5):
        json_structure['shots'].append({
            'query': possible_shots[i][1],
            'answer': json.dumps({label: possible_shots[i][2][clean_label]})
        })
    with open(f'prompts/few_shots/{label}_shots.json', 'w') as fp:
        json.dump(json_structure, fp)
