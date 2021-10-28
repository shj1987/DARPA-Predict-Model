#### Code Structure

- `main/main.ipynb` : main pipeline
- `clean_text_append.py` : dataset preprocessing
- `frame_classification/` : classification models
  - `string_match_append.py` : using string_match method
  - `url2west_append.py` : using WeSTClass method
  - `LeidosBERT/leidos_bert_append.py` : using LEIDOS BERT method
- `collect_append.py` : collect the classification result

#### Input

- `append_news_raw.json.gz` : raw news data
- `gdelt_time_series.json` : GDELT time series
- `cp5-cpec.exogenous.gdelt.events.v1.json.gz` : raw GDELT data
- `frame_classification/LeidosBERT/cpec_frame.bin` : pretrained BERT parameters

#### Output

- `news_results_append.json` : output classification result
- `main/news_{method}_time_series_to_{date}.json` : output news time series
- `main/news_gdelt_{method}_corr_to_{date}.json` : output correlation matrix with gdelt

