#!/bin/bash

n_folds=$1
config_number=$2

for ((i=0; i<n_folds; i++))
do
  iteration=$i
  echo Fold "$iteration"
  python -m spacy train config/config_$config_number.cfg \
  --output trained_models_grid_search/trained_models_cfg_$config_number/output_run_$iteration \
  --paths.train data_splits/run_$iteration/train.spacy \
  --paths.dev data_splits/run_$iteration/dev.spacy \
  --ignore-warnings \
  --gpu-id 0
  python -m spacy evaluate trained_models_grid_search/trained_models_cfg_$config_number/output_run_$iteration/model-best\
   data_splits/test.spacy --gpu-id 0 > results_grid_search/evaluation_results_cfg_$config_number/test_run_$iteration.txt
done