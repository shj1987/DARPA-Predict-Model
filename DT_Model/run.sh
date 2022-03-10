DATA_ROOT="/data/socialsim_data"
DT_ROOT="2021CP6"

mkdir -p "${DT_ROOT}_csv"

python data_proc.py \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -d "1-31" \
    -df "2021CP6" \
    -p "twitter" \
    -corr "$DATA_ROOT/workdir/cp6_twitter_gdelt_corr.json" \
    -gdelt "$DATA_ROOT/workdir/cp6_gdelt_timeseries.json" \
    -ts "$DATA_ROOT/workdir/cp6_twitter_timeseries.json" \
    -pl 21 -tl 345

python run_original.py -n "$DATA_ROOT/cp6_twitter_nodelist.txt" -d "1-31" -df "2021CP6" -p "twitter" -sd "2021-01-11"
python run_lasso.py -n "$DATA_ROOT/cp6_twitter_nodelist.txt" -d "1-31" -df "2021CP6" -p "twitter" -sd "2021-01-11"
