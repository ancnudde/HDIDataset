model=$1

for run in "system" "control"
do
python -W ignore generative_NER.py $model $run true
python -W ignore generative_NER.py $model $run false
done