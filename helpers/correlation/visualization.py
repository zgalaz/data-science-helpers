import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.correlation.computation import compute_correlation
from helpers.utils.validators import \
    validate_args_rank_one_or_one_dimensional, \
    validate_x_y_numpy_array, \
    validate_x_y_observation_count


@validate_x_y_numpy_array
@validate_x_y_observation_count
@validate_args_rank_one_or_one_dimensional
def plot_correlation(X,
                     y,
                     corr_type="pearson",
                     fig_size=(8, 8),
                     fig_show=True,
                     save_as="figure.pdf",
                     x_label="x",
                     y_label="y"):
    """
    Plot the correlation graph

    This function plots the correlation graph showing the correlation between
    features and labels. The type of correlation is set by an input argument
    determining the interpretation of the relationship. It also shows the
    fitted line along with its confidence intervals.

    Parameters
    ----------

    X : numpy array
        1D feature array

    y : numpy array
        1D labels array

    corr_type : str
        Type of correlation to compute

    fig_size : tuple, optional, default (12, 5)
        Size of the figure

    fig_show : bool, optional, default True
        Figure showing switch

    save_as : bool, optional, default "figure.pdf"
        Name of the saved figure (if None, saving skipped)

    x_label : str, optional, default "x"
        Label of the x-axis

    y_label : str, optional, default "y"
        Label of the y-axis

    Raises
    ------

    TypeError
        Raised when X or y is not an instance of np.ndarray

    ValueError
        Raised when X and y have not the same number of rows (observations)
        Raised when args (input arrays) are not rank-one or one-dimensional
    """

    # Create the figure
    fig = plt.figure(figsize=fig_size)

    # Create the axes
    ax = fig.add_subplot(1, 1, 1)

    # Reshape to 1D but non-rank 1 arrays
    if X.ndim == 1:
        X = X.reshape((len(X), 1))
    if y.ndim == 1:
        y = y.reshape((len(y), 1))

    # Select only observations with no missing data
    X_nans = np.isnan(X)
    y_nans = np.isnan(y)

    X = X[np.logical_not(X_nans | y_nans)]
    y = y[np.logical_not(X_nans | y_nans)]

    # Compute the correlation
    r, p = compute_correlation(X, y, corr_type=corr_type)

    # Plot the joint-plot
    sns.regplot(x=X, y=y, fit_reg=True, order=1, truncate=False)

    # Set up the final adjustments
    ax.set_title(r"{}: $\rho = {:.2}$, $p = {:.2}$".format(corr_type.capitalize(), r, p))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()

    # Store the figure
    if save_as:
        plt.savefig(save_as)

    # Show the graph (if enabled)
    if fig_show:
        plt.show()
    else:
        plt.close()
