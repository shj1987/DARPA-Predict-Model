#!/bin/bash
python lr_plus.py -m cp6_twitter_timeseries_to_11_29.json -g cp6_gdelt_timeseries.json -p twitter -t 15 -n 14 -c cp6_twitter_gdelt_corr_to_11_29.json -o twitter_prediction.json
python lr_plus.py -m cp6_youtube_timeseries_to_11_29.json -g cp6_gdelt_timeseries.json -p youtube -t 15 -n 14 -c cp6_youtube_gdelt_corr_to_11_29.json -o youtube_prediction.json