import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from helpers.classification.metrics import classification_metric
from helpers.utils.validators import validate_args_rank_one_or_one_dimensional


@validate_args_rank_one_or_one_dimensional
def plot_classification(X,
                        y,
                        y_true,
                        y_pred,
                        metrics=("acc", "sen", "spe"),
                        fig_size=(12, 5),
                        fig_show=True,
                        save_as="figure.pdf",
                        x_label="x",
                        y_label="y",
                        **plot_kwargs):
    """
    Cross-validate the binary classification model

    This function cross-validates the input classification model using X, y. The
    cross-validation is sed by num_folds and num_repetitions. After the model is
    cross-validated, several metrics are computed.

    Parameters
    ----------

    X : numpy array
        2D feature matrix (rows=observations, cols=features)

    y : numpy array
        1D labels array

    y_true : numpy array
        1D true labels array

    y_pred : numpy array
        1D predicted labels array

    metrics : tuple, optional, default ("acc", "sen", "spe")
        Tuple with classification metrics to compute

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

    ValueError
        Raised when args (input arrays) are not rank-one or one-dimensional
    """

    # Convert the input data to pd.Series
    if not isinstance(X, pd.Series):
        X = pd.Series(X.reshape((len(X), )))
    if not isinstance(y, pd.Series):
        y = pd.Series(y.reshape((len(y), )))
    if not isinstance(y_true, pd.Series):
        y_true = pd.Series(y_true.reshape((len(y_true), )))
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred.reshape((len(y_pred), )))

    # Compute the classification metrics
    computed_metrics = [(metric, round(classification_metric(metric, y_true, y_pred), 2)) for metric in metrics]

    # Prepare the temporary DataFrame
    df = pd.DataFrame({"X": X, "y": y, "y_true": y_true, "y_pred": y_pred, "matches": y_true == y_pred})

    # Create the figure
    fig = plt.figure(figsize=fig_size)

    # Plot the true labels scatter-plot
    ax = fig.add_subplot(1, 2, 1)
    sns.scatterplot(x="X", y="y", hue="y_true", data=df, **plot_kwargs)

    ax.set_title("Ground truth")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.tight_layout()

    # Plot the predicted labels scatter-plot
    ax = fig.add_subplot(1, 2, 2)
    sns.scatterplot(x="X", y="y", hue="y_pred", size="matches", data=df, **plot_kwargs)

    ax.set_title("Predicted ({})".format(" ".join(["{} = {},".format(m, v) for m, v in computed_metrics])))
    ax.set_xlabel(x_label)
    ax.set_ylabel("")
    plt.tight_layout()

    # Store the figure
    if save_as:
        plt.savefig(save_as)

    # Show the graph (if enabled)
    if fig_show:
        plt.show()
    else:
        plt.close()
