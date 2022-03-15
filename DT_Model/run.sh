DATA_ROOT="/data/socialsim_data"
DT_ROOT="2021CP6"

mkdir -p "${DT_ROOT}"
mkdir -p "${DT_ROOT}_csv"

python data_proc.py \
    -n "$DATA_ROOT/cp6_twitter_nodelist.txt" \
    -d "12-20" \
    -df "2021CP6" \
    -p "twitter" \
    -corr "$DATA_ROOT/workdir/cp6_twitter_gdelt_corr.json" \
    -gdelt "$DATA_ROOT/workdir/cp6_gdelt_timeseries.json" \
    -ts "$DATA_ROOT/workdir/cp6_twitter_timeseries.json" \
    -pl 21 -tl 345

python data_proc.py \
    -n "$DATA_ROOT/cp6_youtube_nodelist.txt" \
    -d "12-20" \
    -df "2021CP6" \
    -p "youtube" \
    -corr "$DATA_ROOT/workdir/cp6_youtube_gdelt_corr.json" \
    -gdelt "$DATA_ROOT/workdir/cp6_gdelt_timeseries.json" \
    -ts "$DATA_ROOT/workdir/cp6_youtube_timeseries.json" \
    -pl 21 -tl 345

python run_original.py -n "$DATA_ROOT/cp6_twitter_nodelist.txt" -d "12-20" -df "2021CP6" -p "twitter" -sd "2020-11-30"
python run_original.py -n "$DATA_ROOT/cp6_youtube_nodelist.txt" -d "12-20" -df "2021CP6" -p "youtube" -sd "2020-11-30"
python run_lasso.py -n "$DATA_ROOT/cp6_twitter_nodelist.txt" -d "12-20" -df "2021CP6" -p "twitter" -sd "2020-11-30"
python run_lasso.py -n "$DATA_ROOT/cp6_youtube_nodelist.txt" -d "12-20" -df "2021CP6" -p "youtube" -sd "2020-11-30"
cp "${DT_ROOT}/*" "${DATA_ROOT}/workdir/"
