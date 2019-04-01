from scipy.stats.stats import pearsonr, spearmanr, kendalltau


def compute_correlation(x, y, corr_type="pearson"):
    """
    Compute correlation coefficient

    Correlation coefficient and its p-value is computed. Three types of correlation
    are supported, specifically:
     - Pearson's correlation
     - Spearman's correlation
     - Kendall's correlation

    Parameters
    ----------

    x : numpy array
        1D feature array

    y : numpy array
        1D labels array

    corr_type : str, optional, default "pearson"
        Type of correlation to compute

    Returns
    -------

    Correlation coefficient and its p-value (tuple)

    Raises
    ------

    ValueError
        Raised when the desired correlation type is not supported
    """

    # Define the correlation mapping
    correlations = {
        "pearson": pearsonr,
        "spearman": spearmanr,
        "kendall": kendalltau
    }

    # Compute the correlation
    if corr_type not in correlations:
        raise ValueError("Corr. type {} unsupported, use: {}".format(corr_type, correlations))
    return correlations[corr_type](x, y)
