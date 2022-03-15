#!/bin/bash
DATA_ROOT="/data/socialsim_data"
WORK_DIR="$DATA_ROOT/workdir"

# CORR_GDELT
python lr_plus.py \
    -m "$WORK_DIR/cp6_twitter_timeseries.json" \
    -g "$WORK_DIR/cp6_gdelt_timeseries.json" \
    --nodes "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -t 15 -n 21 \
    -c "$WORK_DIR/cp6_twitter_gdelt_corr.json" \
    -o "$WORK_DIR/twitter_UIUC_LR_PLUS_CORR_GDELT.json"

python lr_plus.py \
    -m "$WORK_DIR/cp6_youtube_timeseries.json" \
    -g "$WORK_DIR/cp6_gdelt_timeseries.json" \
    --nodes "$DATA_ROOT/cp6_youtube_nodelist.txt" \
    -t 15 -n 21 \
    -c "$WORK_DIR/cp6_youtube_gdelt_corr.json" \
    -o "$WORK_DIR/youtube_UIUC_LR_PLUS_CORR_GDELT.json"

# TC ROBERTA
python lr_plus_tc.py \
    -m "$WORK_DIR/cp6_twitter_timeseries.json" \
    -g "$WORK_DIR/Leidos_time_series_to_2021-01-31.json" \
    --nodes "$DATA_ROOT/cp6_twitter_dmg_nodelist.txt" \
    -n 21 \
    -o "$WORK_DIR/twitter_UIUC_LR_PLUS_TC_ROBERTA.json"

python lr_plus_tc.py \
    -m "$WORK_DIR/cp6_youtube_timeseries.json" \
    -g "$WORK_DIR/Leidos_time_series_to_2021-01-31.json" \
    --nodes "$DATA_ROOT/cp6_youtube_dmg_nodelist.txt" \
    -n 21 \
    -o "$WORK_DIR/youtube_UIUC_LR_PLUS_TC_ROBERTA.json"
 # TC ENTROPY ROBERTA
python lr_plus_tc_ent.py \
    -m "$WORK_DIR/cp6_twitter_timeseries.json" \
    -g "$WORK_DIR/Leidos_time_series_to_2021-01-31.json" \
    -z "$WORK_DIR/cp6_zipf_timeseries.json" \
    --nodes "$DATA_ROOT/cp6_twitter_dmg_nodelist.txt" \
    -n 21 \
    -o "$WORK_DIR/twitter_UIUC_LR_PLUS_TC_ENT_ROBERTA.json"

python lr_plus_tc_ent.py \
    -m "$WORK_DIR/cp6_youtube_timeseries.json" \
    -g "$WORK_DIR/Leidos_time_series_to_2021-01-31.json" \
    -z "$WORK_DIR/cp6_zipf_timeseries.json" \
    --nodes "$DATA_ROOT/cp6_youtube_dmg_nodelist.txt" \
    -n 21 \
    -o "$WORK_DIR/youtube_UIUC_LR_PLUS_TC_ENT_ROBERTA.json"

# TC WEST
python lr_plus_tc.py \
    -m "$WORK_DIR/cp6_twitter_timeseries.json" \
    -g "$WORK_DIR/Westclass_time_series_to_2021-01-31.json" \
    --nodes "$DATA_ROOT/cp6_twitter_dmg_nodelist.txt" \
    -n 21 \
    -o "$WORK_DIR/twitter_UIUC_LR_PLUS_TC_WEST.json"

python lr_plus_tc.py \
    -m "$WORK_DIR/cp6_youtube_timeseries.json" \
    -g "$WORK_DIR/Westclass_time_series_to_2021-01-31.json" \
    --nodes "$DATA_ROOT/cp6_youtube_dmg_nodelist.txt" \
    -n 21 \
    -o "$WORK_DIR/youtube_UIUC_LR_PLUS_TC_WEST.json"
 # TC ENTROPY WEST
python lr_plus_tc_ent.py \
    -m "$WORK_DIR/cp6_twitter_timeseries.json" \
    -g "$WORK_DIR/Westclass_time_series_to_2021-01-31.json" \
    -z "$WORK_DIR/cp6_zipf_timeseries.json" \
    --nodes "$DATA_ROOT/cp6_twitter_dmg_nodelist.txt" \
    -n 21 \
    -o "$WORK_DIR/twitter_UIUC_LR_PLUS_TC_ENT_WEST.json"

python lr_plus_tc_ent.py \
    -m "$WORK_DIR/cp6_youtube_timeseries.json" \
    -g "$WORK_DIR/Westclass_time_series_to_2021-01-31.json" \
    -z "$WORK_DIR/cp6_zipf_timeseries.json" \
    --nodes "$DATA_ROOT/cp6_youtube_dmg_nodelist.txt" \
    -n 21 \
    -o "$WORK_DIR/youtube_UIUC_LR_PLUS_TC_ENT_WEST.json"
