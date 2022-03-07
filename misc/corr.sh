#!/bin/bash

DATA_ROOT="/data/socialsim_data"
START_DATE="2020-02-01"
TRAIN_END_DATE="2021-01-11"

python calc_corr.py \
    -pi "$DATA_ROOT/workdir/cp6_twitter_timeseries.json" \
    -ei "$DATA_ROOT/workdir/cp6_gdelt_timeseries.json" \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -g D -s "$START_DATE" -e "$TRAIN_END_DATE" \
    -o "$DATA_ROOT/workdir/cp6_twitter_gdelt_corr.json"
python calc_corr.py \
    -pi "$DATA_ROOT/workdir/cp6_youtube_timeseries.json" \
    -ei "$DATA_ROOT/workdir/cp6_gdelt_timeseries.json" \
    -n "$DATA_ROOT/cp6_youtube_nodelist.txt" \
    -g D -s "$START_DATE" -e "$TRAIN_END_DATE" \
    -o "$DATA_ROOT/workdir/cp6_youtube_gdelt_corr.json"
    