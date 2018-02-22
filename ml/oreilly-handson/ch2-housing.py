import os
import tarfile
import pandas as pd
from six.moves import urllib
import matplotlib.pyplot as plt
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from pandas.plotting import scatter_matrix
from sklearn.utils import check_array
from scipy import sparse

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Transformer mixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, do not try to understand it (yet).
class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encode categorical features as a numeric array.
    The input to this transformer should be a matrix of integers or strings,
    denoting the values taken on by categorical (discrete) features.
    The features can be encoded using a one-hot aka one-of-K scheme
    (``encoding='onehot'``, the default) or converted to ordinal integers
    (``encoding='ordinal'``).
    This encoding is needed for feeding categorical data to many scikit-learn
    estimators, notably linear models and SVMs with the standard kernels.
    Read more in the :ref:`User Guide <preprocessing_categorical_features>`.
    Parameters
    ----------
    encoding : str, 'onehot', 'onehot-dense' or 'ordinal'
        The type of encoding to use (default is 'onehot'):
        - 'onehot': encode the features using a one-hot aka one-of-K scheme
          (or also called 'dummy' encoding). This creates a binary column for
          each category and returns a sparse matrix.
        - 'onehot-dense': the same as 'onehot' but returns a dense array
          instead of a sparse matrix.
        - 'ordinal': encode the features as ordinal integers. This results in
          a single column of integers (0 to n_categories - 1) per feature.
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:
        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories are sorted before encoding the data
          (used categories can be found in the ``categories_`` attribute).
    dtype : number type, default np.float64
        Desired dtype of output.
    handle_unknown : 'error' (default) or 'ignore'
        Whether to raise an error or ignore if a unknown categorical feature is
        present during transform (default is to raise). When this is parameter
        is set to 'ignore' and an unknown category is encountered during
        transform, the resulting one-hot encoded columns for this feature
        will be all zeros.
        Ignoring unknown categories is not supported for
        ``encoding='ordinal'``.
    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting. When
        categories were specified manually, this holds the sorted categories
        (in order corresponding with output of `transform`).
    Examples
    --------
    Given a dataset with three features and two samples, we let the encoder
    find the maximum value per feature and transform the data to a binary
    one-hot encoding.
    >>> from sklearn.preprocessing import CategoricalEncoder
    >>> enc = CategoricalEncoder(handle_unknown='ignore')
    >>> enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])
    ... # doctest: +ELLIPSIS
    CategoricalEncoder(categories='auto', dtype=<... 'numpy.float64'>,
              encoding='onehot', handle_unknown='ignore')
    >>> enc.transform([[0, 1, 1], [1, 0, 4]]).toarray()
    array([[ 1.,  0.,  0.,  1.,  0.,  0.,  1.,  0.,  0.],
           [ 0.,  1.,  1.,  0.,  0.,  0.,  0.,  0.,  0.]])
    See also
    --------
    sklearn.preprocessing.OneHotEncoder : performs a one-hot encoding of
      integer ordinal features. The ``OneHotEncoder assumes`` that input
      features take on values in the range ``[0, max(feature)]`` instead of
      using the unique values.
    sklearn.feature_extraction.DictVectorizer : performs a one-hot encoding of
      dictionary items (also handles string-valued features).
    sklearn.feature_extraction.FeatureHasher : performs an approximate one-hot
      encoding of dictionary items or strings.
    """

    def __init__(self,
                 encoding='onehot',
                 categories='auto',
                 dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):
        """Fit the CategoricalEncoder to X.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_feature]
            The data to determine the categories of each feature.
        Returns
        -------
        self
        """

        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        """Transform X using one-hot encoding.
        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.
        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.
        """
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(
            np.arange(n_samples, dtype=np.int32), n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix(
            (data, (row_indices, column_indices)),
            shape=(n_samples, indices[-1]),
            dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# Downloads the dataset
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Loads the dataset
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


# Splits the dataset into train / test, deterministically by hashing the id.
# Author's note: Recommended while training because it is deterministic
#                even if the dataset is updated
def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def test_set_check(identifier, test_ratio, hash):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


# Splits the dataset into train / test, randomly.
# Author's note:
# - This is not effective while training because you
#   will never see the same data over multiple sessions.
# - You can set the seed to the same value, but this will
#   break when you update the dataset
#   np.random.seed(42)
def split_train_test_randomly(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def verify_stratified_split(data):
    print(data["income_cat"].value_counts() / len(data))


# Uses imputer to replace missing values by median values
def replace_NA_by_median(data):
    imputer = Imputer(strategy="median")
    imputer.fit(data)  # Computes the median values
    # imputer.strategy
    # imputer.statistics_
    # data.median().values
    # Author's note:
    #   fit_transform() does fit() then transform(). can be more efficient
    return pd.DataFrame(imputer.transform(data), columns=data.columns)


def encode_label(data):
    encoder = LabelEncoder()
    return encoder.fit_transform(data)


def encode_1hot_manual(data):
    encoder = LabelEncoder()
    data_encoded = encoder.fit_transform(data)
    # print(data_encoded)
    # print(encoder.classes_)
    encoder = OneHotEncoder()
    return encoder.fit_transform(data_encoded.reshape(-1, 1))


# 1. Loads the dataset
housing = load_housing_data()

# 2 Looking at the dataset
# * To look at first rows:
#     housing.head()
#
# * To look at columns 'count' and 'type:
#     housing.info()
#
# * To look at counts per label in a specific column:
#     housing["ocean_proximity"].value_counts()
# Author's note :
# - some of total_bedrooms column values are missing
#
# * To make an histogram of the columns
#     housing.hist(bins=50, figsize=(20,15))
#     plt.show()
# Author's note :
# - median_income was scaled on a [0,15] interval
# - housing_median_age is capped at 50
# - median_house_value is capped at 500000
# - Fat-tail distribution are harder to learn on

# 3. Creating a test set

# 3 Data manipulation
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# 4 Splits train and test set
# * Alt1
#    housing = housing.reset_index()  # adds 'index' column
#    train_set, test_set = split_train_test_by_id(housing, 0.2, "index")
# * Alt2
#    train_set, test_set = train_test_split(housing, test_size=.2, random_state=42)
# * Alt3
#    train_set, test_set = split_train_test_by_id(housing, 0.2)
# * Alt4
#    housing["id"] = housing["longitude"] * 1000 + housing["latitude"]
#    train_set, test_set = split_train_test_by_id(housing, 0.2, "id")
#    print(len(train_set), "train + ", len(test_set), "test")
# Alt5 - Stratified split over income cat
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

verify_stratified_split(housing)
verify_stratified_split(strat_train_set)
verify_stratified_split(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()  # copy the train set to housing

# 5 Visualizing Geographical data
#
# * Simple scatter plot
#    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
#
# * Colored circle scatter plot
#    housing.plot(
#        kind="scatter", x="longitude", y="latitude", alpha=0.4,
#        s=housing["population"]/100,  # 'size' of circles
#        label="population", figsize=(10, 7),
#        c="median_house_value",  # 'color' of circles
#        cmap=plt.get_cmap("jet"), colorbar=True)
#    plt.legend()
#    plt.show()
#
# * Examining correlation matrix
# corr_matrix = housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)
#
# * Scatter matrix plot of median_house_value against promising variables
# attributes = [
#    "median_house_value", "median_income", "total_rooms", "housing_median_age"
# ]
# scatter_matrix(housing[attributes], figsize=(12, 8))
#
# * Scatter plot of median_house_value vs median_income
# housing.plot(
#     kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
#
pass

# 6 Modify some variables to make them more relevant
# housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing[
#     "total_rooms"]
# housing["population_per_household"] = housing["population"] / housing[
#     "households"]
# * Look again at corr_matrix and scatter matrix
# corr_matrix = housing.corr()
# corr_matrix["median_house_value"].sort_values(ascending=False)
# attributes = [
#     "median_house_value",
#     "median_income",
#     "rooms_per_household",
#     "housing_median_age",
#     "bedrooms_per_room"
# ]
# scatter_matrix(housing[attributes], figsize=(12, 8))
pass

# 7 Start Again: It cleanup time!
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# 7 (MANUAL) Clean districts w/o total_bedrooms
# * Option 1 - Remove only NA
# housing.dropna(subset=["total_bedrooms"])
# * Option 2 - Drop the whole column
# housing.drop("total_bedrooms", axis=1)
# * Option 3 - Set to median value
# median = housing["total_bedrooms"].median()  # option 3 - set to median value
# housing["total_bedrooms"].fillna(median, inplace=True)
# * Option 3.B - Use Imputer
# housing_tr = replace_NA_by_median(housing.drop("ocean_proximity", axis=1))
#
# 7.2 Encode labels
# * Option 1 - Default encoding. Not so good in this case, because
#              Encoded value 1 vs encoded value 2 have no correlations
# housing_cat_encoded = LabelEncoder().fit_transform(housing["ocean_proximity"])
# * Option 2 - One Hot encoding is much better in this case
# housing_cat_1hot = encode_1hot(housing["ocean_proximity"])
# * Option 3 - LabelBinarizer
# housing_cat_1hot = LabelBinarizer(sparse_output=True).fit_transform(
#     housing["ocean_proximity"])
#
# 7.3 add extra information
# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)
#
# 7.4 Scale normalize/standardize numerical parameters with
#     MinMaxScaler/StandardScaler
pass

# 7 (PIPELINE) Putting it all in one through the Pipeline
num_attribs = list(housing.drop("ocean_proximity", axis=1))
cat_attribs = ["ocean_proximity"]
num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', CategoricalEncoder(encoding="onehot-dense"))
])
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
])
housing_num = num_pipeline.fit_transform(housing)
housing_cat = cat_pipeline.fit_transform(housing)
housing_prepared = full_pipeline.fit_transform(housing)

# 8.L Training linear regression model
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# Look how fare's some fitted data
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions: ", lin_reg.predict(some_data_prepared))
print ("Labels: ", list(some_labels))

# Measure mean_squared error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
# Author's note. The model rmse is $68k dollars, clearly unacceptable.
# This is an example of underfitting

# 8.T Trying a more powerful model
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

# Look at this new model rmse
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
# Author's note. Here it is the opposite, the model overfits the data
