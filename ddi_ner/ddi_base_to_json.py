import os
import json
import xml.etree.ElementTree as et
import xmltodict


def get_sentence(sentence_element):
    text = sentence_element.get('text')
    entities = [elt.get('text')
                for elt in sentence_element if elt.get('text') != None]
    return (text, list(set(entities)))


def extract_documents():
    documents = os.listdir('MedLine')
    doc_data = []
    for doc in documents:
        with open(f'MedLine/{doc}') as fp:
            data = fp.read()
            xml_dict = xmltodict.parse(data)
            doc_data.append(xml_dict)
    return doc_data


articles_list = extract_documents()
dataset = []
for article in articles_list:
    document = article['document']
    sentences = document['sentence']
    for sentence in sentences:
        try:
            formatted_entities = []
            base_entities = sentence['entity']
            if isinstance(base_entities, dict):
                elements = base_entities['@charOffset'].split(';')
                for element in elements:
                    formatted_entities.append([int(element.split('-')[0]),
                                               int(element.split('-')[1]),
                                               'Drug'])
            else:
                for entity in base_entities:
                    elements = entity['@charOffset'].split(';')
                    for element in elements:
                        formatted_entities.append([int(element.split('-')[0]),
                                                   int(element.split('-')[1]),
                                                   'Drug'])
            dataset.append(
                {'text': sentence['@text'], 'gold': formatted_entities})
        except Exception as e:
            pass

with open('ddi_corpus_2.json', 'w') as fp:
    json.dump({'dataset': dataset}, fp)
