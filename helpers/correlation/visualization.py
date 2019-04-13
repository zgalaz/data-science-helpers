import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.correlation.computation import compute_correlation
from helpers.utils.validators import \
    validate_df_dataframe, \
    validate_args_rank_one_or_one_dimensional, \
    validate_x_y_numpy_array, \
    validate_x_y_observation_count


@validate_x_y_numpy_array
@validate_x_y_observation_count
@validate_args_rank_one_or_one_dimensional
def plot_correlation(X,
                     y,
                     corr_type="pearson",
                     ax=None,
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

    ax : matplotlib.axes, optional, default None
        Axes to use for the plot (if no axes are provided, a new figure is created)

    fig_size : tuple, optional, default (8, 8)
        Size of the figure

    fig_show : bool, optional, default True
        Figure showing switch

    save_as : str, optional, default "figure.pdf"
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

    # Create the figure and axes if necessary
    if not ax:
        fig = plt.figure(figsize=fig_size if fig_size else (8, 8))
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


@validate_df_dataframe
def plot_correlation_matrix(df, fig_size=(8, 8), fig_show=True, save_as="figure.pdf", **kwargs):
    """
    Plot the correlation matrix

    This function plots the correlation matrix among all columns in <df> (only
    columns with purely numerical data are used. Column names are used as x/y
    tick labels, and placed in the bottom and the left side, respectively.
    By default, the "coolwarm" cmap is used.

    Parameters
    ----------

    df : pandas.DataFrame
        Pandas DataFrame with the data for plotting

    fig_size : tuple, optional, default (8, 8)
        Size of the figure

    fig_show : bool, optional, default True
        Figure showing switch

    save_as : str, optional, default "figure.pdf"
        Name of the saved figure (if None, saving skipped)

    Raises
    ------

    TypeError
        Raised when df is not an instance of pd.DataFrame
    """

    # Create temporary DataFrame with numerical columns only
    df_num = df.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    corr = df_num.corr()

    # Create the figure
    fig = plt.figure(figsize=fig_size if fig_size else (8, 8))

    # Create the axes
    ax = fig.add_subplot(1, 1, 1)

    # Plot the correlation matrix
    fig.colorbar(ax.matshow(corr, cmap=kwargs.get("fig_cmap", "coolwarm"), vmin=-1, vmax=1))

    # Set up the ticks and labels
    ax.set_xticks(np.arange(0, len(df_num.columns), 1))
    ax.set_yticks(np.arange(0, len(df_num.columns), 1))
    ax.set_xticklabels(df_num.columns)
    ax.set_yticklabels(df_num.columns)

    ax.set_title("")
    ax.set_xlabel("")
    ax.set_ylabel("")

    plt.tick_params(top=True, bottom=False, left=True, right=False)
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Store the figure
    if save_as:
        plt.savefig(save_as)

    # Show the graph (if enabled)
    if fig_show:
        plt.show()
