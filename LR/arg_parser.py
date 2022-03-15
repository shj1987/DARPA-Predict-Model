import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run LR Model.")

    # common parameter:
    parser.add_argument('--platform', type=str, default='twitter',
                        help='Switch between twitter and youtube.')
    parser.add_argument('-s', '--start_date', required=True, type=str, 
                        help='The Start Date (format YYYY-MM-DD)')
    parser.add_argument('-e', '--end_date', required=True, type=str, 
                        help='The End Date (format YYYY-MM-DD (Exclusive))')

    #parameter for CP6:
    parser.add_argument('--dev_test_len', type=int, default=21,
                        help='the testing length.')
    parser.add_argument('--input_source', type=str, default='gdelt|newssrc',
                        help='the input type.')
    parser.add_argument('--timeseries_path', type=str, required=True,
                        help='the input timeseries path.')
    parser.add_argument('--exo_path', type=str, required=True,
                        help='the input exogeneous path.')
    parser.add_argument('--ent_path', type=str, required=True,
                        help='the input entropy path.')
    parser.add_argument('--corr_path', type=str, required=True,
                        help='the input correlation path.')
    parser.add_argument('--nodes_path', type=str, required=True,
                        help='the input node list path.')
    parser.add_argument('--file_name', type=str, default='twitter_UIUC_LR_CORR_GDELT.json',
                        help='the output file path')
    

    args = parser.parse_args()
    return args