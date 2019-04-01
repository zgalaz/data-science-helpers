import inspect
import numpy as np
from collections import defaultdict
from sklearn.model_selection import KFold
from helpers.regression.metrics import regression_metric
from helpers.utils.validators import validate_x_y_numpy_array, validate_x_y_observation_count
from helpers.utils.logger import Logger


@validate_x_y_numpy_array
@validate_x_y_observation_count
def cross_validate_regressor(X,
                             y,
                             model,
                             metrics=("mae", "mse", "rmse", "eer"),
                             num_folds=10,
                             num_repetitions=20,
                             seed=42,
                             logger=None):
    """
    Cross-validate the regression model

    This function cross-validates the input regression model using X, y. The
    cross-validation is set by num_folds and num_repetitions. After the model
    is cross-validated, several metrics are computed.

    Parameters
    ----------

    X : numpy array
        2D feature matrix (rows=observations, cols=features)

    y : numpy array
        1D labels array

    model : class that implements fit, and predict methods
        Initialized regression model

    metrics : tuple, optional, default ("mae", "mse", "rmse", "eer")
        Tuple with regression metrics to compute

    num_folds : int, optional, default 10
        Number of cross-validation folds

    num_repetitions : int, optional, default 20
        Number of cross-validation runs

    seed : int, optional, default 42
        Random generator seed

    logger : Logger, optional, default None
        Logger class

    Returns
    -------

    Default dictionary with keys=metric names, vals=metric arrays

    Raises
    ------

    TypeError
        Raised when X or y is not an instance of np.ndarray

    ValueError
        Raised when X and y have not the same number of rows (observations)
    """

    # Prepare the logger
    logger = logger if logger else Logger(inspect.currentframe().f_code.co_name)

    # Prepare the results table for the cross-validation results
    table_cv_data = defaultdict(list)

    # Run the desired number of cross-validation repetitions
    for repetition in range(num_repetitions):
        for train_i, test_i in KFold(n_splits=num_folds, random_state=seed, shuffle=True).split(X):

            # Split the data to train, test sets
            X_train, X_test = X[train_i], X[test_i]
            y_train, y_test = y[train_i], y[test_i]

            try:

                # fit the regressor
                model.fit(X_train, y_train)

                # Evaluate the regressor
                predicted = model.predict(X_test)

                # Encode the labels
                y_true = y_test
                y_pred = predicted

                # Compute the regression metrics
                for metric in metrics:
                    computed = regression_metric(metric, y_true, y_pred, X.shape[2] if X.ndim > 1 else 1)
                    computed = computed if computed and np.isfinite(computed) else None
                    if computed:
                        table_cv_data[metric].append(computed)

            except Exception as e:
                if "Input contains NaN, infinity or a value too large" in str(e):
                    logger.warning("Poor performance detected, skipping current validation fold")
                    continue
                else:
                    logger.exception(e)

    return table_cv_data
