import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run LR Model.")

    parser.add_argument('--challenge', type=str, default='CP6',
                        help='Switch between CP5 and CP6.')

    # common parameter:
    parser.add_argument('--platform', type=str, default='twitter',
                        help='Switch between twitter and youtube.')
    parser.add_argument('--split', type=int, default=126,
                        help='Split between training and test.')
    parser.add_argument('--pred_len', type=int, default=28,
                        help='Length of prediction period.')
    parser.add_argument('--val_len', type=int, default=28,
                        help='Length of validation period.')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Debug mode.')
    parser.add_argument('--output_file', type=str, default='UIUC_BASELINE_DMG_LEIDOS_fix.json',
                        help='Output file name.')
    parser.add_argument('--output_dir', type=str, default='output/',
                        help='Output file name.')
    parser.add_argument('--start_date', type=str, default='2020-03-30',
                        help='start date of training period.')
    parser.add_argument('--end_date', type=str, default='2020-08-31',
                        help='end date of training period.')

    #parameter for CP6:
    parser.add_argument('--cp6_input_source', type=str, default='gdelt',
                        help='the input type.')
    parser.add_argument('--cp6_dev_test_len', type=int, default=21,
                        help='the testing length.')
    parser.add_argument('--cp6_data_path', type=str, default='/data/leidos_extracted/2021CP6/',
                        help='the input data path.')
    parser.add_argument('--cp6_date', type=str, default='01_10',
                        help='the starting date of testing.')
    parser.add_argument('--cp6_file_name', type=str, default='twitter_UIUC_LR_CORR_GDELT.json',
                        help='the output file path')


    #parameter for CP5:
    parser.add_argument('--cp5_eval_nodes', type=str, default='/data/leidos_extracted/2021CP5/cp5_eval_nodes.txt',
                        help='cp5_eval_nodes')
    parser.add_argument('--cp5_other_nodes', type=str, default='/data/leidos_extracted/2021CP5/cp5_other_nodes.txt',
                        help='cp5_other_nodes.')
    parser.add_argument('--twitter_shift', type=bool, default=False,
                        help='Whether to shift.')
    parser.add_argument('--entropy', type=bool, default=False,
                        help='Whether to add entropy time series.')
    parser.add_argument('--condition', type=bool, default=False,
                        help='Whether to add condition.')
    parser.add_argument('--text_path', type=str, default='/data/leidos_extracted/2021CP5/',
                        help='Text path from dmg.')

    parser.add_argument('--data_path', type=str, default='/data/leidos_extracted/2021CP5/',
                        help='Data path from dmg.')
    parser.add_argument('--twitter_input', type=str, default='news_leidos_bert_time_series_to_8_31.json',
                        help='input time series from dmg')
    parser.add_argument('--twitter_corr', type=str, default='news_lotclass_time_series_to_8_31.json',
                        help='input time series from dmg')
    parser.add_argument('--youtube_corr', type=str, default='youtube_acled_corr_to_6_29.json',
                        help='input time series from dmg')
    parser.add_argument('--youtube_input', type=str, default="acled_time_series.json",
                        help='input time series from dmg')
    parser.add_argument('--entropy_input', type=str, default='/data/leidos_extracted/2021CP5/zipf_time_series_to_8_31.json',
                        help='entropy time series input')
    

    #parameter for CP6:



    args = parser.parse_args()
    return args