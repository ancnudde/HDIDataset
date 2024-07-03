model=$1

RED='\033[0;31m'
NC='\033[0m' # No Color

for run in "system" "control"
do
echo "${RED}Running $model inference, $run prompt, few-shot set to true${NC}"
python -W ignore generative_NER.py $model $run true
# echo "${RED}Running $model inference, $run prompt, few-shot set to false${NC}"
# python -W ignore generative_NER.py $model $run false
done