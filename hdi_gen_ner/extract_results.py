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
    json_re = re.compile(r'([^{]+\{[^}]+\})')
    try:
        json_substring = json_re.search(json_string).group(0)
        return ast.literal_eval(json_substring)
    except Exception as e:
        return None


def results_set_to_json(results, label):
    parsed_content = []
    failed_content = []
    for result in results:
        idx = result[0]
        generated_text, gold_standard = result[2]
        try:
            parsed_json = string_to_json(generated_text)
            generated_label = parsed_json[label.replace('_', ' ')]
            if isinstance(generated_label, list):
                generated_label = list(set(generated_label))
            else:
                generated_label = [generated_label]
            parsed_content.append((idx, generated_label, gold_standard))
        except Exception as e:
            parsed_content.append(
                (idx, [], gold_standard))
            failed_content.append((idx, generated_text, gold_standard))
    return parsed_content, failed_content


def compute_confusion_matrix(results):
    confusion_matrix = {'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0}
    for line in results:
        predict = line['generated']
        gold = line['gold']
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
    with open('data/labels.txt', 'r') as fp:
        labels = [elt.strip().replace(' ', '_') for elt in fp.readlines()]
    prompts = ['0_shots', 'few_shots']
    results = {
        prompt: {
        label: {'precision': 0, 'recall': 0,
                'fscore': 0, 'parsable_fraction': 0}
        for label in labels} for prompt in prompts}
    for prompt in prompts:
        for label in labels:
            try:
                data = open_result_file(
                    f'results/{model}/{prompt}/generation_{label}_shots={prompt=='few_shots'}.json')
                parsed, failed = results_set_to_json(data, label)
                n_parsed = len(parsed) - len(failed)
                results[prompt][label]['parsable_fraction'] = np.round(
                    n_parsed / len(data), 2)
                to_compare = []
                for element in parsed:
                    if label.capitalize().replace('_', ' ') in element[2]:
                        parsed_gold = element[2][label.capitalize().replace(
                            '_', ' ')]
                    else:
                        parsed_gold = []
                    generated = element[1]
                    to_compare.append(
                        {'generated': generated, "gold": parsed_gold})
                with open(f'results/{model}/{prompt}/extracted_generation/{label}_comparison.json', 'w') as fp:
                    json.dump(
                        {'extraction_ratio': results[prompt][label]['parsable_fraction'],
                         'to_compare': to_compare},
                        fp)
                confusion = compute_confusion_matrix(to_compare)
                metrics = compute_metrics(confusion)
                for metric, value in metrics.items():
                    results[prompt][label][metric] = np.round(value, 2)
            except ZeroDivisionError:
                pass
            except FileNotFoundError as e:
                print(e)
                pass
            except Exception as e:
                print(e)
                pass
    return results


def get_comparison_dataframe(results):
    dataframes = []
    for prompt in results.keys():
        prompt_df = pd.DataFrame(results[prompt]).T
        prompt_df['prompt'] = prompt
        dataframes.append(prompt_df)
    global_df = pd.concat(dataframes)
    return global_df


if __name__ == '__main__':
    results = summarize_results('mistral')
    x = get_comparison_dataframe(results)
