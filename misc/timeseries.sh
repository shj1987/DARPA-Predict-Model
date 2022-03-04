#!/bin/bash

DATA_ROOT="/data/socialsim_data"
START_DATE="2020-02-01"
END_DATE="2021-02-01"

python preprocess_extracted_gt.py \
    -i "$DATA_ROOT/Challenge/eval-2/extracted_ground_truth/cp6.ea.gte.twitter-youtube.2020-11-30_2020-12-20.json.gz" \
    -s "$START_DATE" -e "$END_DATE" \
    -o "$DATA_ROOT/test/"
python preprocess_extracted_gt.py \
    -i "$DATA_ROOT/Challenge/eval-3/extracted_ground_truth/cp6.ea.gte.twitter-youtube.2020-12-21_2021-01-10.json.gz" \
    -s "$START_DATE" -e "$END_DATE" \
    -o "$DATA_ROOT/test/"

python timeseries_gdelt.py \
    -i "$DATA_ROOT/Exogenous/GDELT/cp6.ea.gdelt.events.v1.json.gz" \
    -g D \
    -s "$START_DATE" -e "$END_DATE" \
    -o "$DATA_ROOT/test/cp6_gdelt_timeseries.json"
python timeseries_acled.py \
    -i "$DATA_ROOT/Exogenous/ACLED/cp6.ea.acled.v1.csv.gz" \
    -g D \
    -s "$START_DATE" -e "$END_DATE" \
    -o "$DATA_ROOT/test/cp6_acled_timeseries.json"
python timeseries_news_src.py \
    -i "$DATA_ROOT/Exogenous/NewsArticles/cp6.ea.newsarticles.training.v1.json" \
    -g D -t 366 \
    -s "$START_DATE" -e "$END_DATE" \
    -o "$DATA_ROOT/test/cp6_newssrc_timeseries.json"

python timeseries_platform.py \
    -i "$DATA_ROOT/test/cp6.ea.gte.twitter-youtube.2020-11-30_2020-12-20.json_twitter.json" \
       "$DATA_ROOT/test/cp6.ea.gte.twitter-youtube.2020-12-21_2021-01-10.json_twitter.json" \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -p "twitter" \
    -s "$START_DATE" -e "$END_DATE" \
    -o "$DATA_ROOT/test/cp6_twitter_timeseries.json" \
    --statout "$DATA_ROOT/test/cp6_twitter_prob.json"
python timeseries_platform.py \
    -i "$DATA_ROOT/test/cp6.ea.gte.twitter-youtube.2020-11-30_2020-12-20.json_youtube.json" \
       "$DATA_ROOT/test/cp6.ea.gte.twitter-youtube.2020-12-21_2021-01-10.json_youtube.json" \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -p "youtube" \
    -s "$START_DATE" -e "$END_DATE" \
    -o "$DATA_ROOT/test/cp6_twitter_timeseries.json" \
    --statout "$DATA_ROOT/test/cp6_twitter_prob.json"
