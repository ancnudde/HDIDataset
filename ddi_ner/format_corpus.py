import json

with open('data/ddi_corpus_ner.json') as fp:
    data = json.load(fp)['dataset']

content = []
for i, elt in enumerate(data):
    entities = elt['gold']
    formatted_entities = []
    for j, entity in enumerate(entities):
        start_offset = int(entity[0])
        end_offset = int(entity[1])
        entity_name = entity[2]
        formatted_entities.append([start_offset, end_offset, entity_name])
    content.append([i, elt['text'], {"entities": formatted_entities}])

with open('data/ddi_ner.json', 'w') as fp:
    json.dump({'dataset': content}, fp)
