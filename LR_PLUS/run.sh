#!/bin/bash
DATA_ROOT="/data/socialsim_data"

python lr_plus.py \
    -m "$DATA_ROOT/workdir/cp6_twitter_timeseries.json" \
    -g "$DATA_ROOT/workdir/cp6_gdelt_timeseries.json" \
    --nodes "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -t 15 -n 21 \
    -c "$DATA_ROOT/workdir/cp6_twitter_gdelt_corr.json" \
    -o "$DATA_ROOT/workdir/cp6_lr_plus_twitter_prediction.json"

python lr_plus.py \
    -m "$DATA_ROOT/workdir/cp6_youtube_timeseries.json" \
    -g "$DATA_ROOT/workdir/cp6_gdelt_timeseries.json" \
    --nodes "$DATA_ROOT/cp6_youtube_nodelist.txt" \
    -t 15 -n 21 \
    -c "$DATA_ROOT/workdir/cp6_youtube_gdelt_corr.json" \
    -o "$DATA_ROOT/workdir/cp6_lr_plus_youtube_prediction.json"
