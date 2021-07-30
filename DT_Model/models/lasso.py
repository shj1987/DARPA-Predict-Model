"""

 linear_regr.py  (author: Anson Wong / git: ankonzoid)

"""
from sklearn.metrics import mean_squared_error
import warnings
def warn(*args, **kwargs):
    pass
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")
warnings.warn = warn

class lasso:

    def __init__(self):
        from sklearn.linear_model import Lasso
        self.model = Lasso(alpha = 0.1, fit_intercept = False, normalize = True, positive = True)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return mean_squared_error(y, y_pred)

