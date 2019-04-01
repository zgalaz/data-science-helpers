from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score


def sensitivity_score(y_true, y_pred):
    """
    Compute classification sensitivity score

    Classification sensitivity (also named true positive rate or recall) measures
    the proportion of actual positives (class 1) that are correctly identified as
    positives. It is defined as follows:

                     TP
    sensitivity = ---------
                   TP + FN

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

    # Compute the sensitivity score
    return recall_score(y_true, y_pred)


def specificity_score(y_true, y_pred):
    """
    Compute classification specificity score

    Classification specificity (also named true negative rate) measures the
    proportion of actual negatives (class 0) that are correctly identified
    as negatives. It is defined as follows:

                     TN
    specificity = ---------
                   TN + FP

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

    # Compute the confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute the specificity score
    return tn / (tn + fp)


def classification_metric(metric, y_true, y_pred):
    """
    Compute classification metric

    Classification metric is computed using a definition dictionary storing
    key, val pairs that define each classification metric function call.
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

    metric : str
        classification metric abbreviation

    y_true : numpy array
        1D labels array of ground truth labels

    y_pred : numpy array
        1D labels array of predicted labels

    Returns
    -------

    Score value (float)
    """

    # Define the classification metrics
    metrics = {
        "mcc": matthews_corrcoef,
        "acc": accuracy_score,
        "sen": sensitivity_score,
        "spe": specificity_score,
        "pre": precision_score,
        "rec": recall_score,
        "f1": f1_score
    }

    # Compute the classification metric
    if metric not in metrics:
        raise ValueError("Unknown classification metric, supported: {}".format(metrics.keys()))
    return metrics[metric](y_true, y_pred)
