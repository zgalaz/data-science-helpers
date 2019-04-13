from helpers.utils.validators import validate_args_numpy_array
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


@validate_args_numpy_array
def remove_effect_of_covariates(X, c, inline=False):
    """
    Remove the effect of covariates from the features

    This function removes the effect of covariates <c> on features in <X>. The effect
    is removed using linear regression model. More specifically, it is hypothesized
    that if there is an effect, it can removed by subtracting the predictions made
    by the linear model trained as X = f(c), i.e. the features without the effect
    of covariates can be expressed as residuals of such model.

    Parameters
    ----------

    X : numpy array
        1D or 2D array of features (rows=observations, cols=features)

    c : numpy array
        1D or 2D array of covariates (rows=observations, cols=covariates)

    inline : bool, optional, default False
        Specify if it is feasible to alter the original DataFrame

    Returns
    -------

    Feature matrix without the effect of covariates (numpy array)

    Raises
    ------

    TypeError
        Raised when X or c is not an instance of np.ndarray
    """

    # Standardize the values of the covariates
    c = StandardScaler().fit_transform(c)

    # Get the DataFrame to work with
    x = X.copy() if not inline else X

    # Prepare the dimensions
    x = make_2_ndim(x)
    c = make_2_ndim(c)

    for i in range(x.shape[1]):

        # Create the linear regression model
        regressor = LinearRegression()

        # fit the regressor (feature = f(covariates))
        regressor.fit(c, x[:, i])

        # Subtract the effect of covariates (feature = residuals = feature - predictions)
        x[:, i] = x[:, i] - regressor.predict(c)

    return X


def make_1_ndim(x):
    return x.flatten("F") if x.ndim > 1 else x


def make_2_ndim(x):
    return x.reshape((x.shape[0], 1 if x.ndim == 1 else x.shape[1]))
