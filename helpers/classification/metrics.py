from sklearn.metrics import confusion_matrix, recall_score


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
