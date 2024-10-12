import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNet, SGDRegressor, LogisticRegression, SGDClassifier, ElasticNet, Ridge, Lasso, RidgeClassifier, RidgeCV, LassoCV, LogisticRegressionCV, ElasticNetCV, RidgeClassifierCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier 
from xgboost import XGBRegressor, XGBClassifier
from sklearn.svm import LinearSVC, SVC, LinearSVR, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


def compare_classifiers(X, y, models=None, metrics=None, average='weighted', cv=3):

    """Compare the performance of the most prominent Scikit-learn classifiers using cross_val_predict and specified metrics.
    This function is designed to determine which models should be further tuned and optimized.
    X: features
    y: target variable
    models: list of classifiers to evaluate
    metrics: list of metrics to evaluate the classifiers (strings)
    cv: number of cross-validation folds"""

    if models is None:
        # default list of classifiers
        models = [
        MultinomialNB(),
        LogisticRegression(random_state=42),
        RidgeClassifier(random_state=42),
        SGDClassifier(random_state=42),
        XGBClassifier(random_state=42),
        DecisionTreeClassifier(random_state=42),
        KNeighborsClassifier(),
        ExtraTreesClassifier(random_state=42),
        RandomForestClassifier(random_state=42),
        AdaBoostClassifier(random_state=42),
        GradientBoostingClassifier(random_state=42)
      ]
        
    if metrics is None:
        metrics = ['accuracy', 'confusion_matrix', 'precision', 'recall', 'f1', 'classification_report', 'pr_auc', 'roc_auc']

    
    for model in models:
        model_name = model.__class__.__name__
        y_pred = cross_val_predict(model, X, y, cv=cv) # get predictions using cross_val_predict
        print(f'{model_name}:\n')
        for metric in metrics:
            if metric == 'f1':
                score = f1_score(y, y_pred, average=average)
            elif metric == 'precision':
                score = precision_score(y, y_pred, average=average)
            elif metric == 'recall':
                score = recall_score(y, y_pred, average=average)
            elif metric == 'classification_report':
                score = classification_report(y, y_pred)
            elif metric == 'confusion_matrix':
                score = confusion_matrix(y, y_pred, normalize='true')
            elif metric == 'accuracy':
                score = accuracy_score(y, y_pred)
            elif metric == 'roc_auc':
                if len(set(y)) > 2:  # multiclass case
                    score = roc_auc_score(y, y_pred, average=average, multi_class='ovr')
                else:
                    score = roc_auc_score(y, y_pred)
            elif metric == 'pr_auc':
                if len(set(y)) > 2:  # multiclass case
                    precision = dict()
                    recall = dict()
                    pr_auc_score = 0
                    for i in range(len(set(y))):
                        precision[i], recall[i], _ = precision_recall_curve(y == i, y_pred == i)
                        pr_auc_score += auc(recall[i], precision[i])
                    score = pr_auc_score / len(set(y))
                else:
                    precision, recall, _ = precision_recall_curve(y, y_pred)
                    score = auc(recall, precision)
            print(f'\n{metric.title()}:\n {score}')
        print('\n')

            
def rmse(y_true, y_pred): # the metric we will use to evaluate the regressors
    return np.sqrt(mean_squared_error(y_true, y_pred))


def compare_regressors(X, y, models=None, cv=3):

    """Evaluate the performance of the most prominent Scikit-learn regressors using cross_val_predict and RMSE
    X: features
    y: target variable
    models: list of regressors to evaluate
    cv: number of cross-validation folds"""

    if models is None:
        # default list of regressors
        models = [
        LinearRegression(),
        Ridge(),
        ElasticNet(),
        SGDRegressor(random_state=42),
        XGBRegressor(random_state=42),
        DecisionTreeRegressor(random_state=42),
        KNeighborsRegressor(),
        ExtraTreesRegressor(random_state=42),
        RandomForestRegressor(random_state=42),
        AdaBoostRegressor(random_state=42),
        GradientBoostingRegressor(random_state=42)
      ]
    
    for model in models:
        model_name = model.__class__.__name__
        y_pred = cross_val_predict(model, X, y, cv=cv)
        score = rmse(y, y_pred)
        print(f'{model_name}: RMSE = {score}')



def features_to_drop_clf(model, X, y, cv=3, average='weighted'):
    """
    Drop each column one by one and see the effect on the model's performance.
    
    Parameters:
    model - the classifier model to evaluate
    X - a dataset (DataFrame)
    y - target variable
    cv - number of cross-validation folds (default is 3)
    average - the type of averaging performed on the data (default is 'binary')
    
    Returns:
    A list of features to remove which have a negative effect on the model's performance.
    """

    # Check if X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    
    # Convert DataFrame to NumPy array
    X_np = X.to_numpy()
    columns = X.columns
    
    # Calculate the initial F1 score with all features
    initial_f1 = f1_score(y, cross_val_predict(model, X_np, y, cv=cv), average=average)
    print(f'Initial F1 score: {initial_f1}')
    to_remove = []
    
    for i, col in enumerate(columns):
        # Drop the current column
        X_copy = np.delete(X_np, i, axis=1)
        
        # Perform cross-validation prediction
        y_pred = cross_val_predict(model, X_copy, y, cv=cv)
        
        # Calculate the F1 score without the current feature
        current_f1 = f1_score(y, y_pred, average=average)
        
        # Print the classification report for debugging
        print(f'Column: {col}')
        print(f'Classification Report:\n {classification_report(y, y_pred)}')
        
        # If the F1 score without the column is greater than or equal to the initial F1 score, mark the column for deletion
        if current_f1 >= initial_f1:
            to_remove.append(col)
    
    print(f'Features to remove: {to_remove}')
    
    # Evaluate the model's performance after removing all identified features
    X_reduced = X.drop(columns=to_remove).to_numpy()
    final_f1 = f1_score(y, cross_val_predict(model, X_reduced, y, cv=cv), average=average)
    print(f'Initial F1 Score: {initial_f1}')
    print(f'Final F1 Score after dropping features: {final_f1}')
    return to_remove


def features_to_drop_reg(model, X, y, cv=3):
    """
    Drop each column one by one and see the effect on the model's performance.
    
    Parameters:
    model - the regressor model to evaluate
    X - a dataset (DataFrame)
    y - target variable
    cv - number of cross-validation folds (default is 3)
    
    Returns:
    A list of features to remove which have a negative effect on the model's performance.
    """

    # Check if X is a DataFrame
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame")
    
    # Convert DataFrame to NumPy array
    X_np = X.to_numpy()
    columns = X.columns
    
    # Calculate the initial RMSE with all features
    initial_rmse = rmse(y, cross_val_predict(model, X_np, y, cv=cv))
    print(f'Initial RMSE: {initial_rmse}')
    to_remove = []
    
    for i, col in enumerate(columns):
        # Drop the current column
        X_copy = np.delete(X_np, i, axis=1)
        
        # Perform cross-validation prediction
        y_pred = cross_val_predict(model, X_copy, y, cv=cv)
        
        # Calculate the RMSE without the current feature
        current_rmse = rmse(y, y_pred)
        
        # If the RMSE without the column is greater than or equal to the initial RMSE, mark the column for deletion
        if current_rmse >= initial_rmse:
            to_remove.append(col)
    
    print(f'Features to remove: {to_remove}')
    
    # Evaluate the model's performance after removing all identified features
    X_reduced = X.drop(columns=to_remove).to_numpy()
    final_rmse = rmse(y, cross_val_predict(model, X_reduced, y, cv=cv))
    print(f'Initial RMSE: {initial_rmse}')
    print(f'Final RMSE after dropping features: {final_rmse}')
    return to_remove