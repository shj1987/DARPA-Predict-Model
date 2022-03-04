import json
import argparse
import dateutil.parser
import gzip
from collections import defaultdict
import logging
import pathlib
import pandas as pd

def convert_datetime(timestr):
    try:
        ret = pd.to_datetime(timestr, unit='s')
    except:
        try:
            ret = pd.to_datetime(timestr, unit='ms')
        except:
            ret = pd.to_datetime(timestr)
    return ret.tz_localize(None)

def main(args):
    filestem = pathlib.Path(args.path).stem
    logging.info(f'Reading {args.path}')
    fps = {}
    if pathlib.Path(args.path).suffix == '.gz':
        f = gzip.open(args.path, 'rt')
    elif pathlib.Path(args.path).suffix == '.json':
        f = open(args.path, 'r')
    else:
        raise ValueError(f'Unknown file type: {datapath}')
    for line in f:
        tmp = json.loads(line)
        if convert_datetime(tmp['nodeTime']) < pd.to_datetime(args.startdate).tz_localize(None):
            continue
        if convert_datetime(tmp['nodeTime']) >= pd.to_datetime(args.enddate).tz_localize(None):
            continue
        if tmp['platform'] not in fps:
            logging.info(f'Found new platform {tmp["platform"]}')
            fps[tmp['platform']] = open(pathlib.Path(args.outpath).joinpath(f'{filestem}_{tmp["platform"]}.json'), 'w')
        fps[tmp['platform']].write(line)
    f.close()
    for k in fps.keys():
        fps[k].close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess extracted groundtruth data.')
    parser.add_argument('-i', '--path', required=True, type=str, help='Path to the extracted groundtruth file(s) (.json|.gz)')
    parser.add_argument('-s', '--startdate', required=True, type=dateutil.parser.isoparse, help='The Start Date (format YYYY-MM-DD)')
    parser.add_argument('-e', '--enddate', required=True, type=dateutil.parser.isoparse, help='The End Date (format YYYY-MM-DD (Exclusive))')
    parser.add_argument('-o', '--outpath', required=True, type=str, help='Path to a folder to save the preprocess files (.json)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    main(args)
