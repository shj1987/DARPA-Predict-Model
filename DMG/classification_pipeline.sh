GPU=0

# Step-1 Clean the input file
INPUT='./data/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json.gz'
INPUT_GDELT='./data/cp6.ea.gdelt.events.v1.json.gz'
CLEAN_OUTPUT='./data/cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'

python clean_text_append.py $INPUT $CLEAN_OUTPUT $INPUT_GDELT

# Step-1 Phrase mining
UCPHRASE_INPUT='./ucphrase/testdata/ucphrase_input_cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
UCPHRASE_DECODE_DIR='./ucphrase/decoded/cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10'
UCPHRASE_DECODE_PATH='./ucphrase/decoded/cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10/decoded/doc2sents-0.85-tokenized.id.ucphrase_input_cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
UCPHRASE_OUTPUT_PATH='./ucphrase/ucphrase_cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'

python ucphrase_prepare_input.py -i ${CLEAN_OUTPUT} -o ${UCPHRASE_INPUT}
python ucphrase/src/predict.py --gpu ${GPU} --path_corpus ${UCPHRASE_INPUT} --dir_output ${UCPHRASE_DECODE_DIR}
python ucphrase/src/postprocess.py --path_cleaned_corpus ${CLEAN_OUTPUT} --path_ucphrase_decoded ${UCPHRASE_DECODE_PATH} --path_output ${UCPHRASE_OUTPUT_PATH}

# Run WestClass
python url2west_append.py 

# Run Pre-trained Roberta
./classification_roberta.sh

# Run Retrieval & Fine-tuned Roberta
./classification_pipeline.sh

# # Run Merge
# python merge.py
