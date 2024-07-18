model=$1

RED='\033[0;31m'
NC='\033[0m' # No Color
# 'ETHNIC_GROUP' 'HERB_NAME' 'DRUG' 'PATHOLOGY' 'AGE' 'FREQUENCY' 'DURATION' 'COHORT' 'TARGET' 'SEX' 'STUDY' 'PARAMETER' 'EXTRACTION_PROCESS' 'HERB_PART' 'AMOUNT'
#  
for run in 'ETHNIC_GROUP' 'HERB_NAME' 'DRUG' 'PATHOLOGY' 'AGE' 'FREQUENCY' 'DURATION' 'COHORT' 'TARGET' 'SEX' 'STUDY' 'PARAMETER' 'EXTRACTION_PROCESS' 'HERB_PART' 'AMOUNT'
do
# echo "${RED}Running $model inference, $run prompt, few-shot set to false${NC}"
# python generative_NER.py $model $run false
echo "${RED}Running $model inference, $run prompt, few-shot set to true${NC}"
python generative_NER.py $model $run true
done
