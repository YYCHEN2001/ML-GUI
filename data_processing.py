import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(file_path):
    """
    Load data from a CSV or Excel file and return the DataFrame along with its columns.

    Parameters:
    - file_path: str, the path to the CSV or Excel file.

    Returns:
    - df: DataFrame, the loaded data.
    - columns: List[str], list of column names in the DataFrame.
    """
    # 判断文件类型并加载数据
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a CSV or Excel file.")

    # 获取列名
    columns = df.columns.tolist()

    return df, columns


def rename_target(df, new_target_column):
    """
    Renames a specified column to 'target' in the DataFrame.

    Parameters:
    - df: DataFrame, the DataFrame where the column needs to be renamed.
    - new_target_column: str, the name of the column to be renamed to 'target'.

    Returns:
    - df: DataFrame, the updated DataFrame with the renamed column.
    """
    if new_target_column in df.columns:
        df.rename(columns={new_target_column: 'target'}, inplace=True)
        print(f"Column '{new_target_column}' renamed to 'target'.")
    else:
        print(f"Error: Column '{new_target_column}' does not exist in the DataFrame.")
    return df


def split_dataset(df, target_column, test_size=0.25, random_state=42):
    """
    Splits the dataset into training and testing sets.

    Parameters:
    - df: DataFrame, the dataset to be split.
    - target_column: str, the name of the target column.
    - test_size: float, proportion of the dataset to include in the test split.
    - random_state: int, controls the shuffling applied to the data before applying the split.

    Returns:
    - X_train: DataFrame, training feature data.
    - X_test: DataFrame, testing feature data.
    - y_train: Series, training target data.
    - y_test: Series, testing target data.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

