GPU=0

DATA_ROOT="/data/socialsim_data"

# Step-1 Clean the input file
INPUT="$DATA_ROOT/Challenge/eval-4/news_articles/cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json.gz"
INPUT_GDELT="$DATA_ROOT/Exogenous/GDELT/cp6.ea.gdelt.events.v1.json.gz"
CLEAN_OUTPUT="$DATA_ROOT/workdir/cleaned_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json"

python clean_text_append.py $INPUT $CLEAN_OUTPUT $INPUT_GDELT

# Step-1 Phrase mining
UCPHRASE_INPUT='./ucphrase/testdata/ucphrase_input_cleaned_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json'
UCPHRASE_DECODE_DIR='./ucphrase/decoded/cleaned_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31'
UCPHRASE_DECODE_PATH='./ucphrase/decoded/cleaned_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31/decoded/doc2sents-0.85-tokenized.id.ucphrase_input_cleaned_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json'
UCPHRASE_OUTPUT_PATH='./ucphrase/ucphrase_cleaned_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json'

python ucphrase_prepare_input.py -i ${CLEAN_OUTPUT} -o ${UCPHRASE_INPUT}
python ucphrase/src/predict.py --gpu ${GPU} --path_corpus ${UCPHRASE_INPUT} --dir_output ${UCPHRASE_DECODE_DIR}
python ucphrase/src/postprocess.py --path_cleaned_corpus ${CLEAN_OUTPUT} --path_ucphrase_decoded ${UCPHRASE_DECODE_PATH} --path_output ${UCPHRASE_OUTPUT_PATH}

# Run WestClass
INPUT_DIR='./ucphrase/'
INPUT_PREFIX='ucphrase_cleaned'
python url2west_append.py $INPUT_DIR $INPUT_PREFIX

# Run Pre-trained Roberta
./classification_roberta.sh

# Run Retrieval & Fine-tuned Roberta
# ./classification_retrieval.sh

# Merge Results of three methods
# Output under v1_append/, change this in consts.py
python merge.py $UCPHRASE_OUTPUT_PATH


#Generate timeseries data
#GDELT DATA is too large to be uploaded
START_DATE='2020-02-01'
END_DATE='2021-01-31'
GDELT_SERIES="$DATA_ROOT/workdir/cp6_gdelt_timeseries.json"
python gen_ts.py $INPUT_GDELT $START_DATE $END_DATE $GDELT_SERIES
