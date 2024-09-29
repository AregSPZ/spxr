import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, LogisticRegression, SGDClassifier, ElasticNet 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier 
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, auc, precision_recall_curve, average_precision_score, roc_auc_score




def compare_classifiers(X, y, metrics, svm=False):
    """Compare the performance of the most prominent Scikit-learn classifiers using cross_val_predict and specified metrics
    X: features
    y: target variable
    metrics: list of metrics to evaluate the classifiers
    svm: boolean, whether to include SVM classifiers"""

    classifiers = [
    LogisticRegression(random_state=42),
    SGDClassifier(random_state=42),
    RandomForestClassifier(random_state=42),
    ExtraTreesClassifier(random_state=42),
    GradientBoostingClassifier(random_state=42),
    XGBClassifier(random_state=42)
    ]

    if svm: 
        classifiers.append(LinearSVC())
        classifiers.append(SVC(kernel='linear'))
        classifiers.append(SVC(kernel='poly'))
        classifiers.append(SVC(kernel='rbf'))   

    for model in classifiers:
        model_name = repr(model)
        y_pred = cross_val_predict(model, X, y, cv=3) # get predictions using cross_val_predict
        results = {metric.__name__: metric(y, y_pred) for metric in metrics}
        results_str = ', '.join([f'{name}={score}' for name, score in results.items()]) # format the results
        print(f'{model_name}: {results_str}')


def rmse(y_true, y_pred): # the metric we will use to evaluate the regressors
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compare_regressors(X, y, svm = False):
    """Evaluate the performance of the most prominent Scikit-learn regressors using cross_val_predict and RMSE"""

    regressors = [
    LinearRegression(),
    ElasticNet(),
    SGDRegressor(random_state=42),
    DecisionTreeRegressor(random_state=42),
    RandomForestRegressor(random_state=42),
    ExtraTreesRegressor(random_state=42),
    HistGradientBoostingRegressor(random_state=42),
    GradientBoostingRegressor(random_state=42),
    AdaBoostRegressor(random_state=42),
    XGBRegressor(random_state=42)
    ]

    if svm: 
        regressors.append(LinearSVR())
        regressors.append(SVR(kernel='linear'))
        regressors.append(SVR(kernel='poly'))
        regressors.append(SVR(kernel='rbf'))


    for model in regressors:
        model_name = repr(model)
        y_pred = cross_val_predict(model, X, y, cv=3)
        score = rmse(y, y_pred)
        print(f'{model_name}: RMSE = {score}')