"""
Use this script to parse results from the generation to valid json content.

The script generates two sets: one with the valid JSON content along with their
gold-sandard from the original dataset, the second with the unparsable generated
text for further analysis.
"""

import os
import re
import ast
import json


def open_result_file(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)['results']
    return data


def string_to_json(json_string):
    """
    Extracts formatted JSON from the text generated by LLM.
    """
    json_re = re.compile(' ?\{(( )?("|\')' + "DRUGS" + '("|\'): .*)*\}')
    try:
        json_substring = json_re.search(json_string).group(0)
        return ast.literal_eval(json_substring)
    except Exception as e:
        print(e)
        return None


def results_set_to_json(results):
    parsed_content = []
    failed_content = []
    error_counter = {'parsable': 0, 'unparsable': 0}
    for result in results:
        idx = result[0]
        generated_text, gold_standard = result[2]
        parsed_json = string_to_json(generated_text)
        if parsed_json:
            parsed_content.append((idx, parsed_json, gold_standard))
        else:
            parsed_content.append((idx, {"DRUGS": []}, gold_standard))
            failed_content.append((idx, generated_text, gold_standard))
    return parsed_content, failed_content


def spans_to_entities(dataset):
    formatted_dataset = []
    for example in dataset:
        idx = example[0]
        text = example[1]
        entities = example[2][1]['entities']
        named_entities = []
        for entity in entities:
            entity_class = text[entity[0]:entity[1] + 1]
            named_entities.append(entity_class)
        formatted_dataset.append([idx, text, named_entities])
    return formatted_dataset


def compute_confusion_matrix(results):
    confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for predict, gold in results:
        for entity in predict:
            if entity in gold:
                confusion_matrix['TP'] += 1
            else:
                confusion_matrix['FP'] += 1
        for std in gold:
            if std not in predict:
                confusion_matrix['FN'] += 1
    return confusion_matrix


def compute_metrics(confusion_matrix):
    metrics = {'precision': 0, 'recall': 0, 'fscore': 0}
    metrics['precision'] = confusion_matrix['TP'] / \
        (confusion_matrix['TP'] + confusion_matrix['FP'])
    metrics['recall'] = confusion_matrix['TP'] / \
        (confusion_matrix['TP'] + confusion_matrix['FN'])
    f1_numerator = 2 * metrics['precision'] * metrics['recall']
    f1_denominator = (metrics['precision'] + metrics['recall'])
    metrics['fscore'] = f1_numerator / f1_denominator
    return metrics


if __name__ == '__main__':
    data = open_result_file(
        'results/phi/generation_system_shots=True.json')
    parsed, failed = results_set_to_json(data)
    entities = spans_to_entities(data)
    to_compare = []
    for generated, gold in zip(parsed, entities):
        gold_entities = list(set([elt.lower() for elt in gold[2]]))
        parsed_json = list(set([elt.lower() for elt in generated[1]['DRUGS']]))
        to_compare.append((parsed_json, gold_entities))
    confusion = compute_confusion_matrix(to_compare)
    metrics = compute_metrics(confusion)
