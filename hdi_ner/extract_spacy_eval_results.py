import re
import json

results = {}

extraction_re = re.compile('(.*?) *([0-9|.]{5}) *([0-9|.]{5}) *([0-9|.]{5})')

for i in range(10):
    with open(f'test_run_{i}.txt', 'r') as fp:
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
