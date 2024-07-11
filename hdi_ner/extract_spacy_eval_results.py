import re
import json
import numpy as np
import pandas as pd

results = {}

extraction_re = re.compile('(.*?) *([0-9|.]{5}) *([0-9|.]{5}) *([0-9|.]{5})')

for i in range(10):
    with open(f'results_grid_search/evaluation_results_cfg_0/test_run_{i}.txt', 'r') as fp:
        data = fp.readlines()
    metrics = {'precision': {}, 'recall': {}, 'fscore': {}}
    results[i] = metrics
    for elt in data:
        extraction = extraction_re.search(elt)
        if extraction:
            results[i]['precision'][extraction.group(
                1)] = float(extraction.group(2))
            results[i]['recall'][extraction.group(
                1)] = float(extraction.group(3))
            results[i]['fscore'][extraction.group(
                1)] = float(extraction.group(4))

with open('results_hdi/ner_results.json', 'w') as fp:
    json.dump(results, fp)

results_to_list = []
for key, val in results.items():
    element = {'index': key}
    for metric, value in val.items():
        element[metric] = value['Drug']
    results_to_list.append(element)

scores_df = pd.DataFrame.from_dict(results_to_list)
scores_df.set_index('index', inplace=True)

mean_scores = {'precision': np.mean(scores_df['precision']),
               'recall': np.mean(scores_df['recall']),
               'F1-score': np.mean(scores_df['fscore'])}

mean_scores_df = pd.DataFrame.from_dict(mean_scores)
