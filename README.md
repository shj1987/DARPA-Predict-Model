# DARPA-Predict-Model
Prediction models for CP5

1. Get article classification (DMG notebooks)
2. Compute timeseries and GDELT correlations using `timeseries.ipynb`
3. Compute ACLED-YouTube correlation using `corr.ipynb`
4. Compute entropy envelope using `entropy.ipynb`
5. Generate previous-user data using `prev_user.ipynb`
6. MODEL GROUPS
    * arima model (lr_plus.py):
        * `-m`: the main data file we use to train
        * `-g`: the gdelt exogenous data file
        * `-p`: which platform we want to predict on
        * `-t`: how many gdelt events we want to use in our model
        * `-n`: how many days we want to predict
        * `-c`: the correlation matrix file
        * `-o`: the output file
   * Decision_tree model (DT_Model folder)
        * `src`: contains the decision tree model structure.
        * `models`: the linear_models for predicion use in decision tree model.
        * `data_proc.py`: preprocessing for the input file. Different processing for different input files are named respectively in the file.
        * `run_MODEL.py`: model training and inferencing file. MODEL variable can be `[original, lasso, lasso_2]` which correspond to the model `[Linear_Regression, LASSO, LASSO_2]` models.
        * `Output_format`: each model outputs three files, top5, top10 and avg, taking the topk and average value of the prediction values.
7. Realize the predicted number to event list using `network_fill.ipynb`
