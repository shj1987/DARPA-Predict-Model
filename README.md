# DARPA-Predict-Model
Prediction models for CP5

1. Get article classification (DMG notebooks)
2. Compute timeseries and GDELT correlations using `timeseries.ipynb`
3. Compute ACLED-YouTube correlation using `corr.ipynb`
4. Compute entropy envelope using `entropy.ipynb`
5. Generate previous-user data using `prev_user.ipynb`
6. MODEL GROUPS
    * arima model (arima_scale.py):
        * `platform`: the platform we want to predict
        * `after_shift_gdelt`: how many days to shift to remove the dissemination lag between topics and gdelt news
        * `top_num`: how many gdelt events to form the input data
        * `test_num`: how many days as the test dates
        * `corrmat_file`: name of the correlation matrix file
        * `main_file`: time series of the chosen platform
7. Realize the predicted number to event list using `network_fill.ipynb`
