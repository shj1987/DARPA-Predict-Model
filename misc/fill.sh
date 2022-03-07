#!/bin/bash

DATA_ROOT="/data/socialsim_data"
START_DATE="2020-02-01"
TRAIN_END_DATE="2021-01-11"
FULL_END_DATE="2021-02-01"

PRED_START_DATE="2021-01-11"
PRED_END_DATE="2021-02-01"

python network_fill.py \
    -i "$DATA_ROOT/workdir/cp6_lr_plus_twitter_prediction.json" "$DATA_ROOT/workdir/cp6_lr_plus_youtube_prediction.json" \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" "$DATA_ROOT/cp6_youtube_nodelist.txt" \
    -pn "$DATA_ROOT/cp6_twitter_preserve_nodelist.txt" "$DATA_ROOT/cp6_youtube_preserve_nodelist.txt" \
    -p "$DATA_ROOT/workdir/cp6_twitter_prob.json" "$DATA_ROOT/workdir/cp6_youtube_prob.json" \
    -g "$DATA_ROOT/workdir/cp6.ea.gte.twitter-youtube.2020-11-30_2020-12-20.json_twitter.json;$DATA_ROOT/workdir/cp6.ea.gte.twitter-youtube.2020-12-21_2021-01-10.json_twitter.json" "$DATA_ROOT/workdir/cp6.ea.gte.twitter-youtube.2020-11-30_2020-12-20.json_youtube.json;$DATA_ROOT/workdir/cp6.ea.gte.twitter-youtube.2020-12-21_2021-01-10.json_youtube.json" \
    -t "uiuc" -m "UIUC_LR_CORR_GDELT" \
    -sp "eval-4" \
    -ts "$START_DATE" -te "$TRAIN_END_DATE" \
    -ps "$PRED_START_DATE" -pe "$PRED_END_DATE" \
    -o "$DATA_ROOT/workdir/cp6_lr_plus_corr_gdelt_filled.json"
