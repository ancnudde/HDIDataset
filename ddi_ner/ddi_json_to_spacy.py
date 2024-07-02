import json
import spacy
from spacy.tokens import DocBin
from spacy.training import offsets_to_biluo_tags, biluo_tags_to_spans
import random
import spacy
from copy import deepcopy
import itertools
import os


def adapt_offsets_to_spacy(all_data):
    """Makes sure the offsets used to locate entities work with spacy format"""
    for i, line in enumerate(all_data):
        entities = line[2]['entities']
        for j, entity in enumerate(entities):
            all_data[i][2]['entities'][j][1] += 1
    return all_data


def format_annotations(dataset, nlp):
    """Transforms examples to Documents format used by spacy"""
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
    """Generates Docbins, the serialization format used by spacy for CLI training"""
    for i, _ in enumerate(folds):
        if not os.path.exists(f'data_splits/run_{i}/'):
            os.makedirs(f'data_splits/run_{i}/')
        if not os.path.exists(f'trained_models/output_run_{i}/'):
            os.makedirs(f'trained_models/output_run_{i}/')
        folds_copy = deepcopy(folds)
        dev = folds_copy.pop(i)
        train = list(itertools.chain.from_iterable(folds_copy))
        # indices of splits, used to keep track of data splits afterwards
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


with open('data/ddi_ner.json') as fp:
    data = json.load(fp)['dataset']


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    data_with_adapted_offsets = adapt_offsets_to_spacy(data)
    split = len(data_with_adapted_offsets) // 10 * 9
    train = data_with_adapted_offsets[:split]
    test = data_with_adapted_offsets[split:]
    folds = generate_folds(train, 10)
    folds_to_docbin(folds, nlp)
    test_to_docbin(test, nlp)
