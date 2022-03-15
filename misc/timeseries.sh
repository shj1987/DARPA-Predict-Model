#!/bin/bash

DATA_ROOT="/data/socialsim_data"
WORK_DIR="$DATA_ROOT/workdir"
START_DATE="2020-02-01"
FULL_END_DATE="2021-02-01" # Exclusive
TRAIN_END_DATE="2020-11-30" # Exclusive

# Extract GT
python preprocess_extracted_gt.py \
    -i "$DATA_ROOT/ExtractedGroundTruth/cp6.ea.extractedgroundtruth.twitter.v1.json.gz" \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -s "$START_DATE" -e "$TRAIN_END_DATE" \
    -o "$WORK_DIR/"
python preprocess_extracted_gt.py \
    -i "$DATA_ROOT/ExtractedGroundTruth/cp6.ea.extractedgroundtruth.youtube.v1.json.gz" \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -s "$START_DATE" -e "$TRAIN_END_DATE" \
    -o "$WORK_DIR/"

# # Get timeseries of exogenous data
python timeseries_gdelt.py \
    -i "$DATA_ROOT/Exogenous/GDELT/cp6.ea.gdelt.events.v1.json.gz" \
    -g D \
    -s "$START_DATE" -e "$FULL_END_DATE" \
    -o "$WORK_DIR/cp6_gdelt_timeseries.json"
python timeseries_acled.py \
    -i "$DATA_ROOT/Exogenous/ACLED/cp6.ea.acled.v1.csv.gz" \
    -g D \
    -s "$START_DATE" -e "$FULL_END_DATE" \
    -o "$WORK_DIR/cp6_acled_timeseries.json"
python timeseries_news_src.py \
    -i "$DATA_ROOT/Exogenous/NewsArticles/cp6.ea.newsarticles.training.v1.json" \
    -g D -t 366 \
    -s "$START_DATE" -e "$FULL_END_DATE" \
    -o "$WORK_DIR/cp6_newssrc_timeseries.json"

# Get timeseries of platform data
python timeseries_platform.py \
    -i "$WORK_DIR/cp6.ea.extractedgroundtruth.twitter.v1.json_twitter.json" \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -p "twitter" \
    -s "$START_DATE" -e $TRAIN_END_DATE \
    -o "$WORK_DIR/cp6_twitter_timeseries.json" \
    --statout "$WORK_DIR/cp6_twitter_prob.json"

python timeseries_platform.py \
    -i "$WORK_DIR/cp6.ea.extractedgroundtruth.youtube.v1.json_youtube.json" \
    -n "$DATA_ROOT/cp6_youtube_nodelist.txt" \
    -p "youtube" \
    -s "$START_DATE" -e $TRAIN_END_DATE \
    -o "$WORK_DIR/cp6_youtube_timeseries.json" \
    --statout "$WORK_DIR/cp6_youtube_prob.json"
