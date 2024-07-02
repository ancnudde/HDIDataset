import json
import spacy
from spacy.tokens import DocBin
from spacy.training import offsets_to_biluo_tags, biluo_tags_to_spans
import random
import spacy
from copy import deepcopy
import itertools
import os


def format_annotations(dataset, nlp):
    documents = []
    for i, text, annotations in dataset:
        try:
            document = nlp(text)
            tags = offsets_to_biluo_tags(document, annotations['entities'])
            entities = biluo_tags_to_spans(document, tags)
            document.ents = entities
            documents.append(document)
        except Exception as e:
            with open('logging/spacy_formatting_errors.txt', 'w') as fp:
                fp.write(f'{e}\n{text}\n{annotations}\n\n')
    return documents


def generate_folds(dataset, n_folds):
    folds = []
    shuffled_dataset = random.sample(dataset, len(dataset))
    fold_size = len(shuffled_dataset) // n_folds
    for i in range(n_folds):
        folds.append(shuffled_dataset[i * fold_size:(i + 1) * fold_size])
    return folds


def folds_to_docbin(folds, nlp):
    for i, _ in enumerate(folds):
        if not os.path.exists(f'data_splits/run_{i}/'):
            os.makedirs(f'data_splits/run_{i}/')
        folds_copy = deepcopy(folds)
        dev = folds_copy.pop(i)
        train = list(itertools.chain.from_iterable(folds_copy))
        dev_idx = [elt[0] for elt in dev]
        train_idx = [elt[0] for elt in train]
        with open(f'data_splits/run_{i}/split_idx.json', 'w') as fp:
            json.dump({'dev': dev_idx, 'train': train_idx}, fp)
        formatted_test = format_annotations(dev, nlp)
        formatted_train = format_annotations(train, nlp)
        dev_db = DocBin()
        dev_db.get_docs(nlp.vocab)
        train_db = DocBin()
        train_db.get_docs(nlp.vocab)
        for line in formatted_test:
            dev_db.add(line)
        dev_db.to_disk(f'data_splits/run_{i}/dev.spacy')
        for line in formatted_train:
            train_db.add(line)
        train_db.to_disk(f'data_splits/run_{i}/train.spacy')


def test_to_docbin(dev_set, nlp):
    data_db = DocBin()
    data_db.get_docs(nlp.vocab)
    formatted_dev = format_annotations(dev_set, nlp)
    for line in formatted_dev:
        data_db.add(line)
        data_db.to_disk(f'data_splits/test.spacy')


if __name__ == '__main__':
    with open('data/hdi_corpus_ner.json') as fp:
        data = json.load(fp)['dataset']
    nlp = spacy.load('en_core_web_sm')
    split = len(data) // 10 * 9
    train = data[:split]
    test = data[split:]
    folds = generate_folds(train, 10)
    folds_to_docbin(folds, nlp)
    test_to_docbin(test, nlp)
