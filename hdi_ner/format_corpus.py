import json

with open('data/hdi_corpus_ner.json') as fp:
    data = json.load(fp)['dataset']

content = []
for i, elt in enumerate(data):
    entities = elt[1]['label']
    formatted_entities = []
    for j, entity in enumerate(entities):
        start_offset = int(entity[0])
        end_offset = int(entity[1])
        entity_name = entity[2]
        formatted_entities.append([start_offset, end_offset, entity_name])
    content.append([i, elt[0], {"entities": formatted_entities}])

with open('data/hdi_ner.json', 'w') as fp:
    json.dump({'dataset': content}, fp)
