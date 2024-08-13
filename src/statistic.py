import pandas as pd
from scipy import stats
import numpy as np

def descriptive_stats_numeric(data, tukey_factor=1.5):
    """
    Calculate descriptive statistics for numeric data.

    Parameters:
    - data (pd.Series): Numeric data for which to calculate statistics.
    - tukey_factor (float, optional): Factor used to calculate the lower and upper fences for outlier detection. Default is 1.5.

    Returns:
    - dict: A dictionary containing various descriptive statistics including count, min, max, mean, standard deviation, 
            median, quartiles, IQR, skewness, kurtosis, outlier fences, and number of outliers.

    Author:
    - Claudia Jara H.

    Date:
    -2024-07-01

    Version:
    - 1.0

    Usage Example:
    - stats = descriptive_stats_numeric(data_series)

    Dependencies:
    - scipy
    - numpy

    Notes:
    - Ensure that the input data is numeric. The function does not handle non-numeric data.
    """
    desc = stats.describe(data)
    percentiles = np.percentile(data, [25, 50, 75])
    q1, q3 = percentiles[0], percentiles[2]
    iqr = q3 - q1
    lower_fence = q1 - tukey_factor * iqr
    upper_fence = q3 + tukey_factor * iqr
    outliers = data[(data < lower_fence) | (data > upper_fence)]

    return {
        "column": data.name,
        "n": desc.nobs,
        "min": desc.minmax[0],
        "max": desc.minmax[1],
        "mean": desc.mean,
        "std": np.sqrt(desc.variance),
        "median": percentiles[1],
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "skewness": desc.skewness,
        "kurtosis": desc.kurtosis,
        "lower_fence": lower_fence,
        "upper_fence": upper_fence,
        "n_outliers": len(outliers)
    }

def descriptive_stats_categorical(data):
    """
    Calculate descriptive statistics for categorical data.

    Parameters:
    - data (pd.Series): Categorical data for which to calculate statistics.

    Returns:
    - dict: A dictionary containing various descriptive statistics including count, number of unique values, mode, mode count,
            second most common value, and its count.

    Author:
    - Claudia Jara H.

    Date:
    -2024-07-01

    Version:
    - 1.0

    Usage Example:
    - stats = descriptive_stats_categorical(data_series)

    Dependencies:
    - pandas

    Notes:
    - The function assumes that the input data is categorical. Numeric data will not be processed correctly.
    """
    value_counts = data.value_counts()
    return {
        "column": data.name,
        "n": len(data),
        "n_unique": data.nunique(),
        "mode": value_counts.idxmax(),
        "mode_count": value_counts.max(),
        "second_most_common": value_counts.index[1] if len(value_counts) > 1 else None,
        "second_most_common_count": value_counts.iloc[1] if len(value_counts) > 1 else None,
    }

def analyze_dataset_numerical(df):
    """
    Analyze numeric columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing numeric columns to be analyzed.

    Returns:
    - pd.DataFrame: A DataFrame containing descriptive statistics for each numeric column.

    Author:
    - Claudia Jara H.

    Date:
    -2024-07-01

    Version:
    - 1.0

    Usage Example:
    - result_df = analyze_dataset_numerical(df)

    Dependencies:
    - pandas
    - scipy
    - numpy

    Notes:
    - Only numeric columns are processed. Non-numeric columns will be ignored.
    """
    results = []  

    for column in df.select_dtypes(include=np.number).columns:
        col_data = df[column].dropna()
        stats = descriptive_stats_numeric(col_data)
        results.append(stats)   

    return pd.DataFrame(results)

def analyze_dataset_categorical(df):
    """
    Analyze categorical columns in the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing categorical columns to be analyzed.

    Returns:
    - pd.DataFrame: A DataFrame containing descriptive statistics for each categorical column.

    Author:
    - Claudia Jara H.

    Date:
    -2024-07-01

    Version:
    - 1.0

    Usage Example:
    - result_df = analyze_dataset_categorical(df)

    Dependencies:
    - pandas

    Notes:
    - Only categorical columns are processed. Numeric columns will be ignored.
    """
    results = []  

    for column in df.select_dtypes(exclude=np.number).columns:
        col_data = df[column].dropna()
        stats = descriptive_stats_categorical(col_data)
        results.append(stats)

    return pd.DataFrame(results)

def overview_df(df):
    """
    Generate an overview of the DataFrame.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be analyzed.

    Returns:
    - pd.DataFrame: A DataFrame providing an overview of each column, including type, number of unique values,
                    missing values, missing percentage, and number of duplicated rows.

    Author:
    - Claudia Jara H.

    Date:
    -2024-07-01

    Version:
    - 1.0

    Usage Example:
    - overview = overview_df(df)

    Dependencies:
    - pandas

    Notes:
    - This function provides general information about each column in the DataFrame.
    """
    df_overview = []

    for column in df.columns:
        types = df[column].dtype
        unique_data = df[column].nunique() 
        missing_count = df[column].isnull().sum()
        value_count = len(df[column])
        missing_percentage = round(missing_count / value_count * 100, 2)
        duplicated = df.duplicated().sum()
        df_overview.append({
            "column": column,
            "types": types,
            "unique_data": unique_data,
            "missing_value": missing_count,
            "missing_percentage": missing_percentage,
            "duplicated": duplicated
        })
    
    df_info = pd.DataFrame(df_overview, columns=['column', 'types', 'unique_data', 'missing_value', 'missing_percentage', 'duplicated'])
    return df_info

def analyze_dataset(df):
    """
    Analyze each column in the dataset, providing both numeric and categorical statistics.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing columns to be analyzed.

    Returns:
    - pd.DataFrame: A DataFrame containing descriptive statistics for each column, including numeric and categorical stats.

    Author:
    - Claudia Jara H.

    Date:
    -2024-07-01

    Version:
    - 1.0

    Usage Example:
    - result_df = analyze_dataset(df)

    Dependencies:
    - pandas
    - scipy
    - numpy

    Notes:
    - This function combines both numeric and categorical statistics into a single DataFrame.
    """
    results = []
    
    for column in df.columns:
        col_data = df[column].dropna()
        if pd.api.types.is_numeric_dtype(df[column]):
            stats = descriptive_stats_numeric(col_data)
        else:
            stats = descriptive_stats_categorical(col_data)
        
        # Overview data
        unique_data = df[column].nunique()
        missing_count = df[column].isnull().sum()
        value_count = len(df[column])
        missing_percentage = round(missing_count / value_count * 100, 2)
        duplicated = df.duplicated().sum()
        
        # Combine all stats into a single dictionary
        combined_stats = {
            "column": column,
            "n": stats.get("n", value_count),
            "min": stats.get("min"),
            "max": stats.get("max"),
            "mean": stats.get("mean"),
            "std": stats.get("std"),
            "median": stats.get("median"),
            "q1": stats.get("q1"),
            "q3": stats.get("q3"),
            "iqr": stats.get("iqr"),
            "skewness": stats.get("skewness"),
            "kurtosis": stats.get("kurtosis"),
            "lower_fence": stats.get("lower_fence"),
            "upper_fence": stats.get("upper_fence"),
            "n_outliers": stats.get("n_outliers"),
            "n_unique": stats.get("n_unique", unique_data),
            "mode": stats.get("mode"),
            "mode_count": stats.get("mode_count"),
            "second_most_common": stats.get("second_most_common"),
            "second_most_common_count": stats.get("second_most_common_count"),
            "unique_data": unique_data,
            "missing_value": missing_count,
            "missing_percentage": missing_percentage,
            "duplicated": duplicated
        }
        
        results.append(combined_stats)
    
    return pd.DataFrame(results)