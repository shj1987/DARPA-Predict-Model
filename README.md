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
   * Decision_tree model (DT_Model folder)
        * `src`: contains the decision tree model structure.
        * `models`: the linear_models for predicion use in decision tree model.
        * `data_proc.py`: preprocessing for the input file. Different processing for different input files are named respectively in the file.
        * `run_MODEL.py`: model training and inferencing file. MODEL variable can be `[original, lasso, lasso_2]` which correspond to the model `[Linear_Regression, LASSO, LASSO_2]` models.
        * `Output_format`: each model outputs three files, top5, top10 and avg, taking the topk and average value of the prediction values.
7. Realize the predicted number to event list using `network_fill.ipynb`
