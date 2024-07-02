import json

with open('warnings.txt') as fp:
    data = fp.readlines()
    warns = [elt[187:237] for elt in data if elt.startswith('WARNING:py')]
    
    with open('data/hdi_corpus_ner.json') as fp:
        data = json.load(fp)['dataset']
   
to_check = []     
for warn in warns:
    for sentence in data:
        if sentence[1].startswith(warn):
            to_check.append(sentence[0])
            
cleaned_data = [elt for i, elt in enumerate(data) if i not in to_check]

for i, elt in enumerate(cleaned_data):
    cleaned_data