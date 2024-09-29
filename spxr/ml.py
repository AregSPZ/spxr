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


def compare_classifiers(X, y, models=[
    LogisticRegression(random_state=42),
    RidgeClassifier(random_state=42),
    SGDClassifier(random_state=42),
    XGBClassifier(random_state=42),
    DecisionTreeClassifier(random_state=42),
    ExtraTreesClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    AdaBoostClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42)], 
    metrics=['accuracy', 'confusion_matrix', 'precision', 'recall', 'f1', 'classification_report', 'pr_auc', 'roc_auc'], cv=3):

    """Compare the performance of the most prominent Scikit-learn classifiers using cross_val_predict and specified metrics.
    This function is designed to determine which models should be further tuned and optimized.
    X: features
    y: target variable
    models: list of classifiers to evaluate
    metrics: list of metrics to evaluate the classifiers (strings)
    cv: number of cross-validation folds"""


    for model in models:
        model_name = model.__class__.__name__
        y_pred = cross_val_predict(model, X, y, cv=cv) # get predictions using cross_val_predict
        print(f'{model_name}:\n')
        for metric in metrics:
            if metric == 'f1':
                score = f1_score(y, y_pred)
            elif metric == 'precision':
                score = precision_score(y, y_pred)
            elif metric == 'recall':
                score = recall_score(y, y_pred)
            elif metric == 'classification_report':
                score = classification_report(y, y_pred)
            elif metric == 'confusion_matrix':
                score = confusion_matrix(y, y_pred, normalize='true')
            elif metric == 'accuracy':
                score = accuracy_score(y, y_pred)
            elif metric == 'roc_auc':
                score = roc_auc_score(y, y_pred)
            elif metric == 'pr_auc':
                precision, recall, _ = precision_recall_curve(y, y_pred)
                score = auc(recall, precision)
            print(f'\n{metric.title()}:\n {score}')
        print('\n')

            
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
    GradientBoostingRegressor(random_state=42)], cv=3):

    """Evaluate the performance of the most prominent Scikit-learn regressors using cross_val_predict and RMSE
    X: features
    y: target variable
    models: list of regressors to evaluate
    cv: number of cross-validation folds"""

    for model in models:
        model_name = model.__class__.__name__
        y_pred = cross_val_predict(model, X, y, cv=cv)
        score = rmse(y, y_pred)
        print(f'{model_name}: RMSE = {score}')



def columns_to_drop_clf(model, X, y, f1, cv=3):
    """
    drop each column one by one and see the effect on the model's performance
    model - the classifier model to evaluate
    X - a dataset (DataFrame) 
    f1 - threshold for the f1 score, typically the actual current f1 score of the model
    returns a list of columns to delete which have a negative effect on the model's performance
    """
    to_delete = []
    for col in X.columns:
        
        X_copy = X.copy()
        X_copy = X_copy.drop(col, axis=1)
        y_train_pred = cross_val_predict(model, X_copy, y, cv=cv)
        print(f'Column: {col}')
        print(f'Classification Report:\n {classification_report(y, y_train_pred)}')
        if f1_score(y, y_train_pred) >= f1:
            to_delete.append(col)
    print(to_delete)