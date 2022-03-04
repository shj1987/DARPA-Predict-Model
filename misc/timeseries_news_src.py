import argparse
import dateutil.parser
import pathlib
import json
import logging

import pandas as pd
import gzip

import tldextract

def extractNewsSrcTimeseries(path, granularity, start_date, end_date, threshold, out_path):
    logging.info('Reading file')
    records = []
    if pathlib.Path(path).suffix == '.gz':
        with gzip.open(path, 'rt') as f:
            for l in f:
                records.append(json.loads(l))
    elif pathlib.Path(path).suffix == '.json':
        with open(path, 'r') as f:
            for l in f:
                records.append(json.loads(l))
    else:
        raise ValueError(f'Unknown file type: {path}')
    
    logging.info('Converting to DataFrame')
    df = pd.DataFrame(records)
    df.date_added = pd.to_datetime(df.date_added)
    df = df[df.date_added.isnull() == False]
    df['press'] = df.url.apply(lambda x: tldextract.extract(x).domain)

    logging.info('Counting')
    news_counts = {
        k: df[df['press'] == k].resample(granularity, on='date_added').press.count().reindex(
            pd.date_range(pd.to_datetime(start_date), pd.to_datetime(end_date) - pd.Timedelta(days=1)), fill_value=0) \
        for k in df.press.unique()
    }
    result_df = pd.DataFrame(news_counts)
    result_df = result_df[list(filter(lambda x: len(x) > 0, result_df.columns))]
    result_df = result_df[dict(sorted(filter(lambda x: x[1] > threshold, result_df.sum(axis=0).to_dict().items()), key=lambda x: -x[1])).keys()]

    srcdf = {}
    for c in result_df.columns:
        srcdf[c] = result_df[c]
    src_json = {k: v.to_json() for k, v in srcdf.items()}
    
    logging.info('Writing to file')
    with open(out_path, 'w') as f:
        f.write(json.dumps(src_json))

def main(args):
    extractNewsSrcTimeseries(args.gdeltpath, args.granularity, args.startdate, args.enddate, args.threshold, args.out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract timeseries from GDELT data.')
    parser.add_argument('-i', '--gdeltpath', required=True, type=str, help='Path to the GDELT file (.json or .json.gz)')
    parser.add_argument('-g', '--granularity', default='D', type=str, choices=['D', 'W'], help='Activity counting granularity (D or W)')
    parser.add_argument('-s', '--startdate', required=True, type=dateutil.parser.isoparse, help='The Start Date (format YYYY-MM-DD)')
    parser.add_argument('-e', '--enddate', required=True, type=dateutil.parser.isoparse, help='The End Date (format YYYY-MM-DD (Exclusive))')
    parser.add_argument('-t', '--threshold', required=True, type=int, help='The threshold to filter out those media sources without the enough posts')
    parser.add_argument('-o', '--out', required=True, type=str, help='Path to save the extract timeseries (.json)')
    args = parser.parse_args()
    for arg in vars(args):
        print(f'{arg} = {getattr(args, arg)}')
    logging.basicConfig(level=logging.INFO)
    main(args)
