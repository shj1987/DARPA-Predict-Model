GPU=0

DATA_ROOT="/data/socialsim_data"
START_DATE="2020-02-01"
FULL_END_DATE="2021-01-31" # Inclusive in this case
CLEAN_OUTPUT="$DATA_ROOT/workdir/cleaned_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json"

ix_name='eval4-twitter'
ix_dir='index_eval4-twitter'
dir_retrieved_docs='./retrieval/retrieved_docs_eval4-twitter'
path_input_cleaned_corpus="$DATA_ROOT/workdir/cleaned_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json"
path_twitter_timeseries_pickle="$DATA_ROOT/workdir/cp6_twitter_timeseries.pkl"

python retrieval/index.py --ix_name ${ix_name} --ix_dir ${ix_dir} --dir_retrieved_docs ${dir_retrieved_docs} --path_input_cleaned_corpus ${path_input_cleaned_corpus} --path_twitter_timeseries_pickle ${path_twitter_timeseries_pickle} --start_date $START_DATE --end_date $FULL_END_DATE

python retrieval/roberta/train.py --gpu ${GPU} --dir_data ${dir_retrieved_docs} --articles ${CLEAN_OUTPUT}


# finetune LeidosRoberta with retrieved docs
# run classification_roberta.sh before running following commands
path_input_csv='./data/eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.csv'
path_output_dir='./roberta/frame'
path_output_csv='./roberta/frame/ft2_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.csv'
path_ft_ckpt='./retrieval/roberta/ft-roberta-epoch2.ckpt'
python roberta/run_roberta.py --input ${path_input_csv} --dir ${path_output_dir} --output ${path_output_csv} --model ${path_ft_ckpt}


# postprocessing
path_frame_names='./roberta/frame/tmp/classes.txt'
path_raw_gz="$DATA_ROOT/Challenge/eval-4/news_articles/cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json.gz"
path_output_id2url='./roberta/id2url_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json'
path_output_append='./roberta/frame/prob_append_ft2_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json'
python roberta/postprocess.py --path_raw_gz ${path_raw_gz} --path_id2url ${path_output_id2url} --path_roberta_output ${path_output_csv} --path_roberta_url2prob ${path_output_append} --path_frame_names ${path_frame_names}