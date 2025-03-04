"""
Use this script to parse results from the generation to valid json content.

The script generates two sets: one with the valid JSON content along with their
gold-sandard from the original dataset, the second with the unparsable generated
text for further analysis.
"""

import re
import ast
import json
import numpy as np
import pandas as pd


def open_result_file(filepath):
    with open(filepath, 'r') as fp:
        data = json.load(fp)['results']
    return data


def string_to_json(json_string):
    """
    Extracts formatted JSON from the text generated by LLM.
    """
    json_re = re.compile(r'[^{]+\{[^}]+\}')
    try:
        json_substring = json_re.search(json_string).group(0)
        return ast.literal_eval(json_substring)
    except Exception as e:
        return None


def results_set_to_json(results):
    parsed_content = []
    failed_content = []
    for result in results:
        idx = result[0]
        generated_text, gold_standard = result[2]
        gold_standard = list(set([entity.lower() for entity in gold_standard]))
        try:
            parsed_json = string_to_json(generated_text)
            generated_drugs = list(set(parsed_json['DRUGS']))
            parsed_content.append((idx, generated_drugs, gold_standard))
        except Exception as e:
            parsed_content.append((idx, {"DRUGS": []}, gold_standard))
            failed_content.append((idx, generated_text, gold_standard))
    return parsed_content, failed_content


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


def summarize_results(model):
    results = {
        'control': {
            'false': {'precision': 0, 'recall': 0, 'fscore': 0, 'parsable_fraction': 0},
            'true': {'precision': 0, 'recall': 0, 'fscore': 0, 'parsable_fraction': 0}},
        'system': {
            'false': {'precision': 0, 'recall': 0, 'fscore': 0, 'parsable_fraction': 0},
            'true': {'precision': 0, 'recall': 0, 'fscore': 0, 'parsable_fraction': 0}}
    }
    for prompt in ['system', 'control']:
        for shots in ['true', 'false']:
            try:
                data = open_result_file(
                    f'results/{model}/generation_{prompt}_shots={shots}.json')
                parsed, failed = results_set_to_json(data)
                n_parsed = len(parsed) - len(failed)
                results[prompt][shots]['parsable_fraction'] = np.round(
                    n_parsed / len(data), 2)
                to_compare = [(parsed_drug[1], parsed_drug[2])
                              for parsed_drug in parsed]
                confusion = compute_confusion_matrix(to_compare)
                metrics = compute_metrics(confusion)
                for metric, value in metrics.items():
                    results[prompt][shots][metric] = np.round(value, 2)
            except ZeroDivisionError:
                pass
    return results


def get_comparison_dataframe(results):
    control_df = pd.DataFrame(results['control']).T
    control_df['prompt'] = 'control'
    system_df = pd.DataFrame(results['system']).T
    system_df['prompt'] = 'system'
    combined_df = pd.concat((system_df, control_df))
    combined_df = combined_df.rename(
        index={'true': 'Few-shots', 'false': '0-shots'})
    return combined_df


if __name__ == '__main__':
    # summary_results_mistral = summarize_results('mistral')
    summary_results_phi = summarize_results('phi')
    # mistral_dataframe = get_comparison_dataframe(summary_results_mistral)
    phi_dataframe = get_comparison_dataframe(summary_results_phi)
