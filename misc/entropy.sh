#!/bin/bash

DATA_ROOT="/data/socialsim_data"
START_DATE="2020-02-01"
FULL_END_DATE="2021-02-01"

python entropy.py \
    -i "$DATA_ROOT/workdir/dmg_0901/merged_hybrid.jsonl" \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -s "$START_DATE" -e "$FULL_END_DATE" \
    -o "$DATA_ROOT/workdir/cp6_zipf_timeseries.json"
    