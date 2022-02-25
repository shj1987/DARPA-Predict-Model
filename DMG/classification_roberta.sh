path_input_json='./data/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json.gz'
path_input_csv='./data/eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.csv'
python roberta/make_data.py --i_path ${path_input_json} --o_path ${path_input_csv}

GPU=4
path_output_dir='./roberta/frame'
path_output_csv='./roberta/frame/LeidosRoberta_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.csv'
python roberta/run_roberta.py --input ${path_input_csv} --dir ${path_output_dir} --output ${path_output_csv}

# postprocessing
path_frame_names='./roberta/frame/tmp/classes.txt'
path_output_id2url='./roberta/id2url_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
path_output_append='./roberta/frame/prob_append_eval3_cp6.ea.newsarticles.youtube.2020-12-21_2021-01-10.json'
python roberta/postprocess.py --path_raw_gz ${path_input_json} --path_id2url ${path_output_id2url} --path_roberta_output ${path_output_csv} --path_roberta_url2prob ${path_output_append} --path_frame_names ${path_frame_names}