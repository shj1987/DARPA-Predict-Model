GPU=0

# Step-1 Clean the input file
INPUT='./data/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json.gz'
INPUT_GDELT='./data/cp6.ea.gdelt.events.v1.json.gz'
CLEAN_OUTPUT='./data/cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'

python clean_text_append.py $INPUT $CLEAN_OUTPUT $INPUT_GDELT

# @Xiaotao: Add your UCPhrase Code
# CLEAN_OUTPUT='/shared/data2/qiz3/socialsim/data/0830appended/cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
UCPHRASE_INPUT='./ucphrase/testdata/ucphrase_input_cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
UCPHRASE_OUTPUT_DIR='./ucphrase/decoded/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10'
python ucphrase_prepare_input.py -i ${CLEAN_OUTPUT} -o ${UCPHRASE_INPUT}
python ucphrase/src/predict.py --gpu ${GPU} --path_corpus ${UCPHRASE_INPUT} --dir_output ${UCPHRASE_OUTPUT_DIR}

# Run WestClass
python url2west_append.py

# # @Xiaotao, Run Retrieval

# # @Xiaotao, Run Roberta


# Run Merge
python merge.py