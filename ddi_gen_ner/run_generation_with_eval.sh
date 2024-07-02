model=$1

for run in "system" "control"
do
python generative_NER.py $model $run true
python generative_NER.py $model $run false
done