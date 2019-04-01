import inspect
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score
from helpers.classification.metrics import sensitivity_score, specificity_score
from helpers.utils.logger import Logger


def cross_validate_classifier(X,
                              y,
                              model,
                              threshold=0.5,
                              metrics=("mcc", "acc", "sen", "spe"),
                              num_folds=10,
                              num_repetitions=20,
                              seed=42,
                              logger=None):
    """
    Cross-validate the binary classification model

    This function cross-validates the input classification model using X, y. The
    cross-validation is sed by num_folds and num_repetitions. After the model is
    cross-validated, several metrics are computed

    Supported classifications metrics:
     - Matthews correlation coefficient (mcc)
     - accuracy (acc)
     - sensitivity (sen)
     - specificity (spe)
     - precision (pre)
     - recall (rec)
     - F1 score (f1)

    Parameters
    ----------

    X : numpy array
        2D feature matrix (rows=observations, cols=features)

    y : numpy array
        1D labels array

    model : class that implements fir, and predict methods
        Initialized binary classification model

    threshold : float, optional, default 0.5
        Threshold for encoding the predicted probability as a class label

    metrics : tuple, optional, default ("mcc", "acc", "sen", "spe")
        Tuple with classification metrics to compute

    num_folds : int, optional, default 10
        Number of cross-validation folds

    num_repetitions : int, optional, default 20
        Number of cross-validation runs

    seed : int, optional, default 42
        Random generator seed

    logger : Logger, optional, default None
        Logger class

    verbose: bool, optional, default False
        Verbose switch

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

    if not all((isinstance(X, np.ndarray), isinstance(y, np.ndarray))):
        raise TypeError('Input arrays must be of type np.ndarray')
    if not X.shape[0] == y.shape[0]:
        raise ValueError('Input arrays must have the same number of rows (observations)')

    # Prepare the logger
    logger = logger if logger else Logger(inspect.currentframe().f_code.co_name)

    # Prepare the classification metrics dict
    cls_metrics = {
        "mcc": matthews_corrcoef,
        "acc": accuracy_score,
        "sen": sensitivity_score,
        "spe": specificity_score,
        "pre": precision_score,
        "rec": recall_score,
        "f1": f1_score
    }

    # Prepare the results table for the cross-validation results
    table_cv_data = defaultdict(list)

    # Prepare counter for poor prediction skips
    num_skips = 0

    # Run the desired number of cross-validation repetitions
    for repetition in range(num_repetitions):

        # Get the cross-validation indices
        kfolds = StratifiedKFold(n_splits=num_folds, random_state=seed, shuffle=True)
        nfolds = 1

        # Cross-validate the classifier
        for train_index, test_index in kfolds.split(X, y):

            try:

                # Split the data to train, test sets
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # fit the classifier
                model.fit(X_train, y_train)

                # Evaluate the classifier
                predicted = model.predict(X_test)

                # Encode the labels
                y_true = pd.Series(y_test, dtype=np.int16)
                y_pred = pd.Series([0 if y_hat < threshold else 1 for y_hat in predicted], dtype=np.int16)

                # Compute the classification metrics
                for metric in metrics:
                    if metric in cls_metrics:
                        table_cv_data[metric].append(cls_metrics[metric](y_true, y_pred))

            except Exception as e:
                if "Input contains NaN, infinity or a value too large" in str(e):
                    logger.warning('Poor performance for {}/{} validation fold'.format(nfolds, num_folds))
                    num_skips += 1
                    continue
                else:
                    logger.exception(e)

            finally:
                nfolds += 1

    # Inform about poor performance skips
    if num_skips > 0:
        logger.info("{}/{} validation folds skipped (performance)".format(num_skips, num_folds * num_repetitions))

    return table_cv_data
