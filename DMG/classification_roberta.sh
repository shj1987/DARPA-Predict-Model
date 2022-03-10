DATA_ROOT="/data/socialsim_data"

path_input_json="$DATA_ROOT/Challenge/eval-4/news_articles/cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json.gz"
path_input_csv="$DATA_ROOT/workdir/eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.csv"
python roberta/make_data.py --i_path ${path_input_json} --o_path ${path_input_csv}

GPU=0
path_output_dir='./roberta/frame'
path_output_csv='./roberta/frame/LeidosRoberta_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.csv'
python roberta/run_roberta.py --input ${path_input_csv} --dir ${path_output_dir} --output ${path_output_csv}

# postprocessing
path_frame_names='./roberta/frame/tmp/classes.txt'
path_output_id2url='./roberta/id2url_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json'
path_output_append='./roberta/frame/prob_append_eval4_cp6.ea.newsarticles.twitter.2021-01-11_2021-01-31.json'
python roberta/postprocess.py --path_raw_gz ${path_input_json} --path_id2url ${path_output_id2url} --path_roberta_output ${path_output_csv} --path_roberta_url2prob ${path_output_append} --path_frame_names ${path_frame_names}
