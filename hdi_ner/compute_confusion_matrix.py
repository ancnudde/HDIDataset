import numpy as np
import pandas as pd
import spacy
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
from spacy.tokens import DocBin
from spacy.training import offsets_to_biluo_tags

nlp = spacy.load(
    'trained_models_grid_search/trained_models_cfg_0/output_run_0/model-best')
matches = {}
test_data = list(DocBin().from_disk(
    'data_splits/test.spacy').get_docs(nlp.vocab))
for line in tqdm(test_data):
    predicted_sentence = nlp(line.text)
    pred_entities = [ent.values()
                     for ent in predicted_sentence.to_json()['ents']]
    gold_ents = [ent.values() for ent in line.to_json()['ents']]
    pred_biluo = offsets_to_biluo_tags(predicted_sentence, pred_entities)
    gold_biluo = offsets_to_biluo_tags(line, gold_ents)
    for predicted, gold in zip(pred_biluo, gold_biluo):
        pair = (predicted, gold)
        if pair in matches:
            matches[pair] += 1
        else:
            matches[pair] = 1


def confusion_counter_to_df(confusion):
    columns = list(set(col[2:] if col != '-' and col !=
                   'O' else col for col, _ in confusion.keys()))
    columns.sort()
    sorted_columns = []
    for col in columns:
        if col != '-' and col != 'O':
            sorted_columns.append('U-' + col)
            sorted_columns.append('B-' + col)
            sorted_columns.append('I-' + col)
            sorted_columns.append('L-' + col)
        else:
            sorted_columns.append(col)
    confusion_frame = pd.DataFrame(
        index=sorted_columns, columns=sorted_columns)
    for (col, row), value in confusion.items():
        confusion_frame.at[row, col] = value
    confusion_frame = confusion_frame.fillna(0)
    return confusion_frame


def plot_confusion_complete_plt(matrix):
    cm = np.array(matrix.values)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[np.newaxis, :]
    data_masked = np.ma.masked_where(cm_normalized == 0, cm_normalized)
    x = list(matrix.columns)
    y = list(matrix.columns)
    fig, ax = plt.subplots(figsize=(15, 15))
    im = ax.imshow(data_masked, cmap=mpl.colormaps['bwr'])
    ax.set_xticks(np.arange(len(x)), labels=x)
    ax.set_yticks(np.arange(len(y)), labels=y)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    plt.grid()
    for i in range(len(x)):
        for j in range(len(y)):
            text = ax.text(j, i, cm[i, j],
                           ha="center", va="center", color="w")
    ax.set_title("Confusion matrix")
    fig.tight_layout()
    plt.show()


counts_df = confusion_counter_to_df(matches)
plot_confusion_complete_plt(counts_df)
