import json
import random


random.seed(47)

with open('data/hdi_extractive_entity_recognition.json', 'r') as fp:
    data = json.load(fp)['dataset']

random.shuffle(data)
split_idx = [sentence[0] for sentence in data][:900]

with open('used_subset_idx.txt', 'w') as fp:
    for idx in split_idx:
        fp.write(f'{idx}\n')
