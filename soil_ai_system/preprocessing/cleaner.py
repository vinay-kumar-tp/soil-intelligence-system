import numpy as np
from sklearn.impute import KNNImputer
from scipy import stats


def handle_missing_values(df):
    """Impute missing numeric values using KNN.

    Args:
        df (pandas.DataFrame): Input dataset.

    Returns:
        pandas.DataFrame: Dataset with imputed numeric values.
    """
    imputer = KNNImputer(n_neighbors=5)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df


def remove_outliers_iqr(df, cols):
    """Remove outliers using the IQR rule for selected columns.

    Args:
        df (pandas.DataFrame): Input dataset.
        cols (list[str]): Columns to apply IQR filtering on.

    Returns:
        pandas.DataFrame: Filtered dataset with outliers removed.
    """
    for col in cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        df = df[(df[col] >= q1 - 1.5 * iqr) & (df[col] <= q3 + 1.5 * iqr)]
    return df.reset_index(drop=True)


def remove_outliers_zscore(df, cols, threshold=3):
    """Remove outliers using z-score thresholding.

    Args:
        df (pandas.DataFrame): Input dataset.
        cols (list[str]): Columns to apply z-score filtering on.
        threshold (float): Z-score cutoff for filtering.

    Returns:
        pandas.DataFrame: Filtered dataset with outliers removed.
    """
    for col in cols:
        df = df[np.abs(stats.zscore(df[col])) < threshold]
    return df.reset_index(drop=True)
