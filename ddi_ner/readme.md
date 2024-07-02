# Named Entity Recognition

`run_training_and_eval.sh` runs the training process and evaluates the performances based. Evaluation uses a k-fold validation; each fold is tested on an independant test-set after training on dev set. The script is used by passing:
- the number of folds as first argument
- the number associated with the config file (0 to 3) as second argument

```console
run_training_and_eval.sh n_fold config_number
```

`run_eval` only runs the evaluation process on pre-generated results files. The same arguments as the previous script are used here: 
- the number of folds as first argument
- the number associated with the config file (0 to 3) as second argument

```console
run_eval.sh n_fold config_number
```

The python script `extract_spacy_eval_results` give a recipe on extracting and processing results of the evaluation.