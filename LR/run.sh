#!/bin/bash

DATA_ROOT="/data/socialsim_data"
WORK_DIR="$DATA_ROOT/workdir"

PRED_START_DATE="2020-11-30"
PRED_END_DATE="2020-12-21" # Exclusive

python LR.py \
    --platform "twitter" \
    --dev_test_len 21 \
    -s $PRED_START_DATE -e $PRED_END_DATE \
    --input_source "gdelt" \
    --timeseries_path "$WORK_DIR/cp6_twitter_timeseries.json" \
    --exo_path "$WORK_DIR/cp6_gdelt_timeseries.json" \
    --ent_path "$WORK_DIR/cp6_zipf_timeseries.json" \
    --corr_path "$WORK_DIR/cp6_twitter_gdelt_corr.json" \
    --nodes_path "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    --file_name "$WORK_DIR/twitter_UIUC_LR_CORR_ENT_GDELT.json"

python LR.py \
    --platform "youtube" \
    --dev_test_len 21 \
    -s $PRED_START_DATE -e $PRED_END_DATE \
    --input_source "gdelt" \
    --timeseries_path "$WORK_DIR/cp6_youtube_timeseries.json" \
    --exo_path "$WORK_DIR/cp6_gdelt_timeseries.json" \
    --ent_path "$WORK_DIR/cp6_zipf_timeseries.json" \
    --corr_path "$WORK_DIR/cp6_youtube_gdelt_corr.json" \
    --nodes_path "$DATA_ROOT/cp6_youtube_nodelist.txt" \
    --file_name "$WORK_DIR/youtube_UIUC_LR_CORR_ENT_GDELT.json"
