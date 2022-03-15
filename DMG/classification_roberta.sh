DATA_ROOT="/data/socialsim_data"

path_input_json="$DATA_ROOT/Exogenous/NewsArticles/cp6.ea.newsarticles.training.v1.json.gz"
path_input_csv="$DATA_ROOT/workdir/cp6.ea.newsarticles.training.csv"
python roberta/make_data.py --i_path ${path_input_json} --o_path ${path_input_csv}

GPU=0
path_output_dir='./roberta/frame'
path_output_csv='./roberta/frame/LeidosRoberta_cp6.ea.newsarticles.training.csv'
python roberta/run_roberta.py --input ${path_input_csv} --dir ${path_output_dir} --output ${path_output_csv}

# postprocessing
path_frame_names='./roberta/frame/tmp/classes.txt'
path_output_id2url='./roberta/id2url_cp6.ea.newsarticles.training.json'
path_output_append='./roberta/frame/prob_append_cp6.ea.newsarticles.training.json'
python roberta/postprocess.py --path_raw_gz ${path_input_json} --path_id2url ${path_output_id2url} --path_roberta_output ${path_output_csv} --path_roberta_url2prob ${path_output_append} --path_frame_names ${path_frame_names}
