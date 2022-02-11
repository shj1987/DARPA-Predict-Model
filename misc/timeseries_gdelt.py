import argparse
import dateutil.parser
import pathlib
import json
import logging

import pandas as pd
import gzip
from tqdm import tqdm

def extractGdeltTimeseries(path, granularity, start_date, end_date, out_path):
    logging.info('Reading file')
    records = []
    if pathlib.Path(path).suffix == '.gz':
        with gzip.open(path, 'r') as f:
            for l in f:
                records.append(json.loads(l))
    else:
        with open(path, 'r') as f:
            for l in f:
                records.append(json.loads(l))
    
    logging.info('Converting to DataFrame')
    alleventcodes = set()
    for r in tqdm(records):
        alleventcodes.add(r['EventCode'])
    alleventcodes = sorted(alleventcodes)
    df = pd.DataFrame(records)
    df.day = pd.to_datetime(df.day)
    df = df.sort_values('day')

    logging.info('Counting')
    eventCountTimeseries = {}
    idxs = pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) - pd.Timedelta(days=1))
    for code in tqdm(alleventcodes, total=len(alleventcodes)):
        tmp = df.query('EventCode == "{}"'.format(code))
        counts = tmp.day.value_counts().resample(granularity).sum()
        counts = counts[pd.to_datetime(start_date):(pd.to_datetime(end_date))].reindex(idxs, fill_value=0)
        eventCountTimeseries[code] = counts

    eventCountTimeseries_json = {k: v.to_json() for k, v in eventCountTimeseries.items()}
    
    logging.info('Writing to file')
    with open(out_path, 'w') as f:
        f.write(json.dumps(eventCountTimeseries_json))

def main(args):
    extractGdeltTimeseries(args.gdeltpath, args.granularity, args.startdate, args.enddate, args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract timeseries from GDELT data.')
    parser.add_argument('-i', '--gdeltpath', required=True, type=str, help='Path to the GDELT file (.json or .json.gz)')
    parser.add_argument('-g', '--granularity', default='D', type=str, choices=['D', 'W'], help='Activity counting granularity (D or W)')
    parser.add_argument('-s', '--startdate', required=True, type=dateutil.parser.isoparse, help='The Start Date (format YYYY-MM-DD)')
    parser.add_argument('-e', '--enddate', required=True, type=dateutil.parser.isoparse, help='The End Date (format YYYY-MM-DD (Exclusive))')
    parser.add_argument('-o', '--out', required=True, type=str, help='Path to save the extract timeseries (.json)')
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    logging.basicConfig(level=logging.INFO)
    main(args)
