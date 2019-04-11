import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import mannwhitneyu
from helpers.common.visualization import starify_pval
from helpers.utils.validators import validate_x_y_str, validate_df_dataframe


@validate_x_y_str
def plot_box_violin(x,
                    y,
                    df,
                    ax=None,
                    fig_size=(8, 8),
                    fig_show=True,
                    save_as="figure.pdf",
                    title=None,
                    x_label=None,
                    y_label=None,
                    violinplot_kwargs=None,
                    boxplot_kwargs=None,
                    stripplot_kwargs=None):
    """
    Plot the box-violin graph

    This function plots the box-violin graph shoving the distribution of values
    for the categories. More specifically, the distribution of <x> over all of
    the categories of <y> in <df> is shown. This graph combines violin plot,
    box plot, and strip plot to improve the visual quality of the graph.

    Parameters
    ----------

    x : str
        Column-name for the categorical variable (x axis)

    y : str
        Column-name for the continuous variable (y axis)

    df : pandas.DataFrame
        Pandas DataFrame with the data for plotting

    ax : matplotlib.axes, optional, default None
        Axes to use for the plot (if no axes, a new figure is created)

    fig_size : tuple, optional, default (8, 8)
        Size of the figure

    fig_show : bool, optional, default True
        Figure showing switch

    save_as : str, optional, default "figure.pdf"
        Name of the saved figure (if None, saving skipped)

    title : str, optional, default None ("")
        Title of the plot

    x_label : str, optional, default None ("")
        Label of the x-axis

    y_label : str, optional, default None ("")
        Label of the y-axis

    violinplot_kwargs : dict, optional, default None
        Kwargs for violin plot

    boxplot_kwargs : dict, optional, default None
        Kwargs for box plot

    stripplot_kwargs : dict, optional, default None
        Kwargs for strip plot

    Raises
    ------

    TypeError
        Raised when X or y is not an instance of str
    """

    # TODO: introduce additional kwargs for finer adjustments

    # Get unique categories for categorical variable defines by y
    categories = list(df[x].unique())

    # Define color scheme
    color_names = ["greens", "purples", "pinks", "browns", "blues", "reds"]
    light_colors = ["#7DCEA0", "#C39BD3", "#FFC9EC", "#E5B699", "#85C1E9", "#D98880"]
    dark_colors = ["#229954", "#884EA0", "#FB84D1", "#B25116", "#2E86C1", "#A93226"]

    # Update color scheme according to the categories
    if len(categories) > len(color_names):
        multiplier = int(len(categories) / len(color_names)) + 1
        color_names = color_names * multiplier
        light_colors = light_colors * multiplier
        dark_colors = dark_colors * multiplier

    # Prepare the final color scheme
    colors = list(zip(color_names, light_colors, dark_colors))

    # Get the colors for the categories
    light_palette = {categories[i]: colors[i][1] for i in range(len(categories))}
    dark_palette = {categories[i]: colors[i][2] for i in range(len(categories))}

    # Default violinplot kwargs
    default_violinplot_kwargs = {
        "palette": light_palette,
        "alpha": 0.2,
        "width": 0.45,
        "inner": None,
        "linewidth": 0,
        "scale": "count",
        "saturation": 0.75,
        "hue_order": categories
    }

    # Default boxplot kwargs
    default_boxplot_kwargs = {
        "boxprops": {"edgecolor": "k", "linewidth": 0},
        "medianprops": {"color": "k", "linewidth": 2},
        "whiskerprops": {"color": "k", "linewidth": 2},
        "capprops": {"color": "k", "linewidth": 2},
        "palette": dark_palette,
        "width": 0.075,
        "fliersize": 0,
        "showcaps": True,
        "whis": 1.5,
        "notch": False,
        "hue_order": categories
    }

    # Default stripplot kwargs
    default_stripplot_kwargs = {
        "palette": light_palette,
        "linewidth": 0,
        "size": 6,
        "alpha": 0.2,
        "split": True,
        "jitter": True,
        "hue_order": categories
    }

    # Prepare kwargs for each plot type
    violinplot_kwargs = violinplot_kwargs if violinplot_kwargs else default_violinplot_kwargs
    boxplot_kwargs = boxplot_kwargs if boxplot_kwargs else default_boxplot_kwargs
    stripplot_kwargs = stripplot_kwargs if stripplot_kwargs else default_stripplot_kwargs

    # Create the figure and axes if necessary
    if not ax:
        fig = plt.figure(figsize=fig_size if fig_size else (8, 8))
        ax = fig.add_subplot(1, 1, 1)

    # Plot the graphs
    sns.violinplot(x=x, y=y, data=df, ax=ax, **violinplot_kwargs)
    sns.stripplot(x=x, y=y, data=df, ax=ax, **stripplot_kwargs)
    sns.boxplot(x=x, y=y, data=df, ax=ax, **boxplot_kwargs)

    # Get feature values for the categories
    observations = [df[df[x] == c] for c in categories]

    # Get values for pairs of the categories
    testing_combinations = combinations(range(len(observations)), 2)

    # Prepare the visualization counter
    count_lines = 0

    # Quantify and visualize the difference between the categories
    for combination in testing_combinations:

        # Get the feature values
        a = observations[combination[0]][y].values
        b = observations[combination[1]][y].values

        # Compute Mann-Whitney U-test
        _, p = mannwhitneyu(a, b)

        # Get the axes limits
        y_min, y_max = ax.get_ylim()

        # Add the visualization of the p-value
        if p <= 0.05:

            # Increment the number of drawn lines
            count_lines += 1

            # Prepare the line drawing partition
            partition = 0.05 * (y_max - y_min)

            # Depict the p-value
            p_str = starify_pval(p)

            # Plot the line connecting the two categories
            y_line_pos = y_max - (count_lines * partition)
            x_line_min = combination[0]
            x_line_max = combination[1]
            ax.hlines(y_line_pos, x_line_min, x_line_max, lw=2.0, color='k')

            # Plot the p-value
            # TODO: improve the adjustment of the y pos
            x_pos_text = (combination[1] + combination[0]) / 2
            y_pos_text = y_line_pos - partition * 5 / 4
            plt.text(x_pos_text, y_pos_text, p_str, horizontalalignment='center', fontsize=24)

    # Apply additional adjustments
    plt.setp(ax.collections, alpha=0.65)

    ax.set_title(title if title else "")
    ax.set_xlabel(x_label if x_label else "")
    ax.set_ylabel(y_label if y_label else "")

    ax.yaxis.grid(True)
    ax.xaxis.grid(True)

    # Store the figure
    if save_as:
        plt.savefig(save_as)

    # Show the graph (if enabled)
    if fig_show:
        plt.show()


@validate_df_dataframe
def plot_missing_values(df,
                        ax=None,
                        fig_size=(8, 8),
                        fig_show=True,
                        save_as="figure.pdf",
                        x_ticklabels=None,
                        y_ticklabels=None,
                        title=None):
    """
    Plot missing values

    This function plots the DataFrame as a 2D plane with missing values (NaNs) as
    black squares and non-missing values (no matter what value) as white squares
    given the ticklabels (if not provided, df.columns (x axis), and df.index (y
    axis) are used by default).

    Parameters
    ----------

    df : pandas.DataFrame
        Pandas DataFrame with the data for plotting

    ax : matplotlib.axes, optional, default None
        Axes to use for the plot (if no axes, a new figure is created)

    fig_size : tuple, optional, default (8, 8)
        Size of the figure

    fig_show : bool, optional, default True
        Figure showing switch

    save_as : str, optional, default "figure.pdf"
        Name of the saved figure (if None, saving skipped)

    x_ticklabels : list or tuple, optional, default None (df.index)
        Labels of the ticks for the x axis (observation names)

    y_ticklabels : list or tuple, optional, default None (df.columns)
        Labels of the ticks for the y axis (feature names)

    title : str, optional, default None ("Missing values")
        Title of the plot

    Raises
    ------

    TypeError
        Raised when df is not an instance of pd.DataFrame
    """

    # Prepare the missing values
    missing_values = df.isnull()

    # Create the figure and axes if necessary
    if not ax:
        fig = plt.figure(figsize=fig_size if fig_size else (8, 8))
        ax = fig.add_subplot(1, 1, 1)

    # Plot the graph
    sns.heatmap(missing_values, ax=ax, cbar=False, cmap="Greys", linewidths=0.3, linecolor="#c8d6e5")

    # Set up the ticks and labels
    ax.set_xticks(np.arange(0, len(df.columns), 1))
    ax.set_yticks(np.arange(0, len(df.index), 1))
    ax.set_xticklabels(x_ticklabels if x_ticklabels else df.columns)
    ax.set_yticklabels(y_ticklabels if y_ticklabels else df.index)
    ax.set_title(title if title else "Missing values")
    ax.grid()

    plt.tick_params(top=False, bottom=True, left=True, right=False)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Store the figure
    if save_as:
        plt.savefig(save_as)

    # Show the graph (if enabled)
    if fig_show:
        plt.show()
