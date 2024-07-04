model=$1

RED='\033[0;31m'
NC='\033[0m' # No Color

for run in 'ALL' 'ETHNIC_GROUP' 'PATHOLOGY' 'AGE' 'FREQUENCY' 'DURATION' 'HERB_NAME' 'COHORT' 'TARGET' 'SEX' 'STUDY' 'DRUG' 'PARAMETER' 'EXTRACTION_PROCESS' 'HERB_PART' 'AMOUNT'
do
echo "${RED}Running $model inference, $run prompt, few-shot set to false${NC}"
python generative_NER.py $model $run false
done