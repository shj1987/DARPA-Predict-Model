#!/bin/bash
python timeseries_gdelt.py -i "cp6.ea.gdelt.events.v1.json.gz" -g D -s "2020-02-01" -e "2021-02-01" -o "cp6_gdelt_timeseries.json"
python timeseries_acled.py -i "cp6.ea.acled.v1.csv.gz" -g D -s "2020-02-01" -e "2021-02-01" -o "cp6_acled_timeseries.json"
python timeseries_news_src.py -i "cp6.ea.newsarticles.training.v1.json" -g D -s "2020-02-01" -e "2021-02-01" -t 366 -o "cp6_newssrc_timeseries.json"
python timeseries_platform.py -i "cp6_twitter_groundtruth_11_30_to_12_20.json" "cp6_twitter_groundtruth_12_21_to_01_10.json" -n "cp6_twitter_nodelist.txt" -s "2020-02-01" -e "2021-02-01" -o "cp6_twitter_timeseries_test.json" -p "cp6_twitter_prob_test.json"
