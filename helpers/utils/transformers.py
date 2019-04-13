import pandas as pd
from helpers.utils.validators import validate_args_numpy_array
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.base import BaseEstimator, TransformerMixin


class CovariateController(BaseEstimator, TransformerMixin):
    """
    Remove (control for) the effect of covariates

    CovariateController controls for the effect of covariates (commonly called
    confounding variables) using linear regression. More specifically, it is
    hypothesized that if there is an effect, it can removed by subtracting the
    predictions made by the linear model trained as X = f(c), i.e. the features
    without the effect of covariates can be expressed as the residuals of such
    model.

    Implemented as a Scikit-learn transformer. Can be fitted on 1D or xD data
    as there can be multiple covariates having an effect on the variables in
    X (features). It assumes that data pre-processing such as feature scaling
    or normalization is done prior using CovariateController.

    The procedure of transforming a feature can be defined as follows:

      [1.]      [2.]              [3.]                            [4.]
    feature = feature - regressor.fit(covariates, feature).predict(covariates)

    [1.] updated feature (residuals of the model)
    [2.] original feature
    [4.] predictions
    [3.] trained linear model feature = f(covariates)

    Parameters
    ----------

    inline : bool, optional, default False
        Parameter that specifies if the transformation should be performed
        inline, i.e. if it is feasible to alter the original DataFrame

    Attributes
    ----------

    regressors : dict, len(models) == len(X.columns) (after fitted)
        Dictionary with the fitted linear regression models for each column
        in X (each column represents a feature). After __init__(), the dict
        is empty. After fit(): {"f1": model, "f2": model, "f3": model, ...}

    covariates : numpy array
        Numpy array storing the covariates used to fit the linear regression
        models. After __init__(), the covariates are unknown and therefore
        the variable is None, After fit(), covariates.shape corresponds to
        the number of observations (rows) and features (cols) used to fit
        the models (model per input feature vector).

    Examples
    --------

    >>> import numpy as np
    >>> import pandas as pd
    >>> from helpers.utils.transformers import CovariateController
    >>>
    >>> # Generate random features and covariates
    >>> f = np.random.randn(1000, 100)
    >>> c = np.random.randn(1000, 3)
    >>> i = 0
    >>>
    >>> # Prepare the DataFrames
    >>> df_f = pd.DataFrame({str(i): f[:, i] for i in range(f.shape[1])})
    >>> df_c = pd.DataFrame({str(i): f[:, i] for i in range(c.shape[1])})
    >>>
    >>> # Control for the effect of covariates
    >>> df_f = CovariateController(inline=True).fit_transform(df_f, df_c)
    >>> df_f.head()

    See also
    --------

    :func:`helpers.utils.transformers.remove_effect_of_covariates`
        Standalone function that removes controls for the effect of covariates
        using pure numpy arrays (not restricted to pd.DataFrames).

    Notes
    -----

    1. Class does not handle missing values (prior transformation required)
    2. Linear regression model used (most common, other models are possible)

    """

    def __init__(self, inline=False):

        # Parse input arguments
        self.inline = inline

        # Prepare instance attributes
        self.regressors = {}
        self.covariates = None

    def fit(self, X, y, **params):
        """
        Fit the transformer

        This method uses the covariates to fit the linear regression models according
        to: features = f(covariates) (model per feature; column of X), i.e. it fits
        the models to capture the (linear) relationship that the covariates and each
        of the features have. The regressor can be modified using input **params.

        Parameters
        --------

        X : numpy array
            1D or 2D array of features (rows=observations, cols=features)

        y : numpy array
            1D or 2D array of covariates (rows=observations, cols=covariates)
        """

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.DataFrame)
        assert X.shape[0] == y.shape[0]

        # Fit the linear regressor (feature = f(covariates))
        self.regressors = {c: LinearRegression(**params).fit(y.values, X[c].values) for c in X.columns}

        # Store the covariates
        self.covariates = y

        return self

    def transform(self, X):
        """
        Transform the data

        This method uses the fitted linear regression models to predict the values of
        the features (columns of X) and replace each of the features by the residuals
        of such model, i.e. feature = feature - feature_hat. With this approach, the
        new feature values are transformed in a way it removes the (linear) effect
        of the covariates on the features.

        Parameters
        --------

        X : numpy array
            1D or 2D array of features (rows=observations, cols=features)

        Returns
        -------

        Feature matrix without the effect of covariates (numpy array)
        """

        assert isinstance(X, pd.DataFrame)

        # Get the DataFrame to work with
        x = X.copy() if not self.inline else X

        # Remove the effect of covariates
        for c in x.columns:
            x[c] = x[c].values - self.regressors.get(c).predict(self.covariates.values)

        return x


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


class TypeSelector(BaseEstimator, TransformerMixin):

    def __init__(self, dtype):
        self.dtype = dtype

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])


class ColumnSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X[self.columns]


def make_1_ndim(x):
    return x.flatten("F") if x.ndim > 1 else x


def make_2_ndim(x):
    return x.reshape((x.shape[0], 1 if x.ndim == 1 else x.shape[1]))
