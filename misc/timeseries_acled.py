import argparse
import dateutil.parser
import pathlib
import json
import logging
from io import StringIO

import pandas as pd
import gzip
from tqdm import tqdm

def extractAcledTimeseries(path, granularity, start_date, end_date, out_path):
    logging.info('Reading file')
    records = []
    if pathlib.Path(path).suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            acled = pd.read_csv(StringIO(f.read()))
    elif pathlib.Path(path).suffix == '.csv':
        with open(path, 'r') as f:
            acled = pd.read_csv(StringIO(f.read()))
    else:
        raise ValueError(f'Unknown file type: {path}')
    
    logging.info('Counting')
    acled.event_date = pd.to_datetime(acled.event_date)
    acled = acled.sort_values('event_date')
    acledCountTimeseries = {}
    idxs = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) - pd.Timedelta(days=1))
    for code in tqdm(acled.event_type.unique(), total=acled.event_type.nunique()):
        tmp = acled.query('event_type == "{}"'.format(code))
        counts = tmp.event_date.value_counts().resample(granularity).sum()
        counts = counts[pd.to_datetime(start_date):(pd.to_datetime(end_date))].reindex(idxs, fill_value=0)
        acledCountTimeseries['_'.join(code.split())] = counts

    acledCountTimeseries_json = {k: v.to_json() for k, v in acledCountTimeseries.items()}
    
    logging.info('Writing to file')
    with open(out_path, 'w') as f:
        f.write(json.dumps(acledCountTimeseries_json))

def main(args):
    extractAcledTimeseries(args.acledpath, args.granularity, args.startdate, args.enddate, args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract timeseries from GDELT data.')
    parser.add_argument('-i', '--acledpath', required=True, type=str, help='Path to the ACLED file (.csv or .csv.gz)')
    parser.add_argument('-g', '--granularity', default='D', type=str, choices=['D', 'W'], help='Activity counting granularity (D or W)')
    parser.add_argument('-s', '--startdate', required=True, type=dateutil.parser.isoparse, help='The Start Date (format YYYY-MM-DD)')
    parser.add_argument('-e', '--enddate', required=True, type=dateutil.parser.isoparse, help='The End Date (format YYYY-MM-DD (Exclusive))')
    parser.add_argument('-o', '--out', required=True, type=str, help='Path to save the extract timeseries (.json)')
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    logging.basicConfig(level=logging.INFO)
    main(args)
