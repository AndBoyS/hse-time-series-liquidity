from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from tslib.scoring import get_score, get_score_estimator


def calibrate_model(model, params, X, y):

    model = GridSearchCV(model(),
                         params,
                         scoring=get_score_estimator,
                         cv=TimeSeriesSplit(n_splits=5),
                         n_jobs=-1)
    model.fit(X, y)

    return model
