#!/bin/bash

DATA_ROOT="/data/socialsim_data"
WORK_DIR="$DATA_ROOT/workdir"
START_DATE="2020-02-01"
TRAIN_END_DATE="2020-11-30" # Exclusive

python calc_corr.py \
    -pi "$WORK_DIR/cp6_twitter_timeseries.json" \
    -ei "$WORK_DIR/cp6_gdelt_timeseries.json" \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -g D -s "$START_DATE" -e "$TRAIN_END_DATE" \
    -o "$WORK_DIR/cp6_twitter_gdelt_corr.json"
python calc_corr.py \
    -pi "$WORK_DIR/cp6_youtube_timeseries.json" \
    -ei "$WORK_DIR/cp6_gdelt_timeseries.json" \
    -n "$DATA_ROOT/cp6_youtube_nodelist.txt" \
    -g D -s "$START_DATE" -e "$TRAIN_END_DATE" \
    -o "$WORK_DIR/cp6_youtube_gdelt_corr.json"
