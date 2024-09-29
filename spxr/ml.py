import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, LogisticRegression, SGDClassifier, ElasticNet, Ridge, Lasso, RidgeClassifier, RidgeCV, LassoCV, LogisticRegressionCV, ElasticNetCV, RidgeClassifierCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier 
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def compare_classifiers(X, y, metrics, models=[
    LogisticRegression(random_state=42),
    RidgeClassifier(random_state=42),
    SGDClassifier(random_state=42),
    XGBClassifier(random_state=42),
    DecisionTreeClassifier(random_state=42),
    ExtraTreesClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    AdaBoostClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42)], cv=3):

    """Compare the performance of the most prominent Scikit-learn classifiers using cross_val_predict and specified metrics.
    This function is designed to determine which models should be further tuned and optimized.
    X: features
    y: target variable
    models: list of classifiers to evaluate
    metrics: list of metrics to evaluate the classifiers"""


    for model in models:
        model_name = model.__class__.__name__
        y_pred = cross_val_predict(model, X, y, cv=cv) # get predictions using cross_val_predict
        results = {metric.__name__: metric(y, y_pred) for metric in metrics if callable(metric)}
        
        # Calculate precision-recall AUC if specified
        if 'pr_auc_score' in [metric.__name__ for metric in metrics if callable(metric)]:
            precision, recall, _ = precision_recall_curve(y, y_pred)
            pr_auc_score = auc(recall, precision)
            results['pr_auc_score'] = pr_auc_score
        
        results_str = ', '.join([f'{name}={score}' for name, score in results.items()]) # format the results
        print(f'{model_name}: {results_str}')


def rmse(y_true, y_pred): # the metric we will use to evaluate the regressors
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compare_regressors(X, y, models=
    [LinearRegression(),
    Ridge(),
    ElasticNet(),
    SGDRegressor(random_state=42),
    XGBRegressor(random_state=42),
    DecisionTreeRegressor(random_state=42),
    ExtraTreesRegressor(random_state=42),
    RandomForestRegressor(random_state=42),
    AdaBoostRegressor(random_state=42),
    GradientBoostingRegressor(random_state=42),
    ],
    cv=3):

    """Evaluate the performance of the most prominent Scikit-learn regressors using cross_val_predict and RMSE
    X: features
    y: target variable
    models: list of regressors to evaluate"""

    for model in models:
        model_name = model.__class__.__name__
        y_pred = cross_val_predict(model, X, y, cv=cv)
        score = rmse(y, y_pred)
        print(f'{model_name}: RMSE = {score}')