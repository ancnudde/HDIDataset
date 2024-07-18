import json


with open('data/ddi_ner.json', 'r') as fp:
    data = json.load(fp)['dataset']

print(len(data))
ents = 0
for elt in data:
    entities = elt[2]['entities']
    for i in entities:
        ents += 1
print(ents)
