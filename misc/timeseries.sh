#!/bin/bash
# python timeseries_gdelt.py -i "/home/dsun18/sstest/cp6.ea.gdelt.events.v1.json.gz" -g D -s "2020-02-01" -e "2021-01-31" -o "/home/dsun18/sstest/cp6_gdelt_timeseries.json"
python timeseries_acled.py -i "/home/dsun18/sstest/cp6.ea.acled.v1.csv.gz" -g D -s "2020-02-01" -e "2021-01-31" -o "/home/dsun18/sstest/cp6_acled_timeseries.json"
# python timeseries_news_src.py -i "/home/dsun18/sstest/cp6.ea.newsarticles.training.v1.json" -g D -s "2020-02-01" -e "2021-01-31" -t 366 -o "/home/dsun18/sstest/cp6_newssrc_timeseries.json"
