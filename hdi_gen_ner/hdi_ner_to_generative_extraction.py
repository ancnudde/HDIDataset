"""
Use this script to transform classical NER dataset to Extractive form.

The entities format is change from (span_start, span_end, entity_class) to 
simply contain the names of the show the name of the corresponding substring
in the text.
"""

import json


def spans_to_entities(dataset):
    formatted_dataset = []
    for example in dataset:
        idx = example[0]
        text = example[1]
        entities = example[2]['entities']
        named_entities = {}
        for entity in entities:
            entity_text = text[entity[0]:entity[1]]
            entity_class = entity[2]
            if entity_class in named_entities:
                named_entities[entity_class].append(entity_text)
            else:
                named_entities[entity_class] = [entity_text]
        formatted_dataset.append([idx, text, named_entities])
    return formatted_dataset


if __name__ == '__main__':
    with open('data/hdi_ner.json', 'r') as fp:
        data = json.load(fp)['dataset']
    formatted_dataset = spans_to_entities(data)
    with open('data/hdi_extractive_entity_recognition.json', 'w') as fp:
        json.dump({'dataset': formatted_dataset}, fp)
