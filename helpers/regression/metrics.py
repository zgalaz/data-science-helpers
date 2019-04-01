import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def root_mean_squared_error(y_true, y_pred):
    """
    Compute root mean squared regression score

    Root mean squared error represents the square root of the sum of the squared
    differences between true and predicted labels, i.e. rmse(mse). It is defined
    as follows:

    rmse = sqrt(mse)
    mse  = sum(y_true - y_hat)^2)

    where y_true is the ground truth label vector, and y_pred is the vector of
    predicted labels.

    Parameters
    ----------

    y_true : numpy array
        1D labels array of ground truth labels

    y_pred : numpy array
        1D labels array of predicted labels

    Returns
    -------

    Score value (float)
    """

    # Compute root mean squared error
    return np.sqrt(mean_squared_error(y_true, y_pred))


def estimation_error_rate(y_true, y_pred):
    """
    Compute estimation error rate score

    Estimation error rate represents the mean absolute error computed between true
    and predicted labels, expressed as a percentage, i.e. mae / range(y_true). It
    is defined as follows:

    eer = mae / range(y_true)
    mae = sum(abs(y_true - y_hat)))

    where y_true is the ground truth label vector, and y_pred is the vector of
    predicted labels.

    Parameters
    ----------

    y_true : numpy array
        1D labels array of ground truth labels

    y_pred : numpy array
        1D labels array of predicted labels

    Returns
    -------

    Score value (float)
    """

    # Compute estimation error rate
    return mean_absolute_error(y_true, y_pred) / np.ptp(y_true)


def adjusted_r_squared(y_true, y_pred, num_predictors=1):
    """
    Compute adjusted r-squared score

    R-squared adjusted represents r-squared adjusted for the number of predictors
    in the model (it increases only if the new term improves the model more than
    would be expected by chance. It decreases when a predictor improves the model
    by less than expected by chance. It is defined as follows:

                                 n - 1
    r2_adj = 1 - (1 - r2^2) * -----------
                               n - p - 1

    where where p is the total number of predictors (not including the constant
    term), and n is the sample size.

    Parameters
    ----------

    y_true : numpy array
        1D labels array of ground truth labels

    y_pred : numpy array
        1D labels array of predicted labels

    num_predictors : int, optional, default 1
        Number of predictors

    Returns
    -------

    Score value (float)
    """

    # Compute adjusted r squared
    return 1 - (((1 - r2_score(y_true, y_pred)) * (len(y_true) - 1)) / (len(y_true) - num_predictors - 1))


def regression_metric(metric, y_true, y_pred, num_predictors=1):
    """
    Compute regression metric

    Regression metric is computed using a definition dictionary storing key,
    val pairs that define each regression metric function call. Supported
    regression metrics:
     - mean absolute error (mae)
     - mean squared error (mse)
     - root mean squared error (mse)
     - estimation error rate (eer)
     - r2, i.e. coefficient of variation (r2)
     - r2 adjusted (r2_adj)

    Parameters
    ----------

    metric : str
        regression metric abbreviation

    y_true : numpy array
        1D labels array of ground truth labels

    y_pred : numpy array
        1D labels array of predicted labels

    num_predictors : int, optional, default 1
        Number of predictors

    Returns
    -------

    Score value (float)
    """

    # Define the regression metrics
    metrics = {
        "mae": mean_absolute_error,
        "mse": mean_squared_error,
        "rmse": root_mean_squared_error,
        "eer": estimation_error_rate,
        "r2": r2_score,
        "r2_adj": adjusted_r_squared
    }

    # Compute the regression metric
    if metric not in metrics:
        raise ValueError("Unknown regression metric, supported: {}".format(metrics.keys()))
    else:
        if metric == "r2_adj":
            return metrics[metric](y_true, y_pred, num_predictors)
        else:
            return metrics[metric](y_true, y_pred)
