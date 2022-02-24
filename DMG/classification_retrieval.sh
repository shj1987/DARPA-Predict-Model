GPU=4

ix_name='eval3-youtube'
ix_dir='index_eval3-youtube'
dir_retrieved_docs='./retrieval/retrieved_docs_eval3-youtube'
path_input_cleaned_corpus='./data/cleaned_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
path_twitter_timeseries_pickle='../data/timeseries/cp6_twitter_timeseries.pkl'

python retrieval/index.py --ix_name ${ix_name} --ix_dir ${ix_dir} --dir_retrieved_docs ${dir_retrieved_docs} --path_input_cleaned_corpus ${path_input_cleaned_corpus} --path_twitter_timeseries_pickle ${path_twitter_timeseries_pickle}

python retrieval/roberta/train.py --gpu ${GPU} --dir_data ${dir_retrieved_docs} 


# finetune LeidosRoberta with retrieved docs
# run classification_roberta.sh before running following commands
path_input_csv='./data/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.csv'
path_output_dir='./roberta/frame'
path_output_csv='./roberta/frame/ft2_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.csv'
path_ft_ckpt='./retrieval/roberta/ft-roberta-epoch2.ckpt'
python roberta/run_roberta.py --input ${path_input_csv} --dir ${path_output_dir} --output ${path_output_csv} --model ${path_ft_ckpt}


# postprocessing
path_raw_gz='./data/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json.gz'
path_output_id2url='./roberta/id2url_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
path_output_append='./roberta/frame/prob_append_ft2_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
python roberta/postprocess.py --path_raw_gz ${path_raw_gz} --path_id2url ${path_output_id2url} --path_roberta_output ${path_output_csv} --path_roberta_url2prob ${path_output_append}