from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV


def search(search_type, x_data, y_data, data_splits, model_type, grid):
    """ Identifying best parameters"""
    results_dict = {}
    if model_type == "logistic_regression":
        # lr = LogisticRegression(max_iter=500, solver='saga', random_state=8)
        lr = LogisticRegression(max_iter=500, solver='saga', random_state=8)
        scoring = {'f1': 'f1', 'loss': 'neg_log_loss'}
        if search_type == 'random':
            logreg_grid = RandomizedSearchCV(estimator=lr,
                                             param_distributions=grid,
                                             cv=data_splits,
                                             scoring=scoring,
                                             verbose=3,
                                             refit='f1',
                                             error_score=0,
                                             return_train_score=True,
                                             random_state=8)

        if search_type == 'grid':
            logreg_grid = GridSearchCV(estimator=lr,
                                       param_grid=grid,
                                       cv=data_splits,
                                       scoring=scoring,
                                       verbose=3,
                                       refit='f1',
                                       error_score=0,
                                       return_train_score=True)
        logreg_grid.fit(x_data, y_data)
        results_dict = {#'best_estimator': logreg_grid.best_estimator_,
                        'best_params': logreg_grid.best_params_,
                        'performance_results': logreg_grid.cv_results_}

    elif model_type == "random_forest":
        rf = RandomForestClassifier()

        if search_type == 'random':
            rf_grid = RandomizedSearchCV(estimator=rf,
                                         param_distributions=grid,
                                         n_iter=30,
                                         cv=data_splits,
                                         scoring='f1',
                                         verbose=3,
                                         refit=False,
                                         error_score=0,
                                         return_train_score=True,
                                         random_state=8)

        if search_type == 'grid':
            rf_grid = GridSearchCV(estimator=rf,
                                   param_grid=grid,
                                   cv=data_splits,
                                   scoring='f1',
                                   verbose=3,
                                   refit=False,
                                   error_score=0,
                                   return_train_score=True)
        rf_grid.fit(x_data, y_data)
        results_dict = {#'best_estimator': rf_grid.best_estimator_,
                        'best_params': rf_grid.best_params_,
                        'performance_results': rf_grid.cv_results_}

    return results_dict
