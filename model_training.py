from sklearn.kernel_ridge import KernelRidge


def train_krr_model(X_train, y_train, params):
    """
    Trains a Kernel Ridge Regression model using the provided training data and parameters.

    Parameters:
    - X_train: DataFrame, feature data for training.
    - y_train: Series, target data for training.
    - params: dict, parameters for the KRR model.

    Returns:
    - model: trained KRR model.
    """
    model = KernelRidge(alpha=params.get('alpha', 1.0),
                        kernel=params.get('kernel', 'linear'),
                        gamma=params.get('gamma', None),
                        degree=params.get('degree', 3),
                        coef0=params.get('coef0', 1))
    model.fit(X_train, y_train)
    return model


def generate_predictions(model, X_train, X_test, y_train, y_test):
    """
    Generate predictions using the trained model and return dataframes containing
    the features, actual values, and predicted values for both training and testing sets.

    Parameters:
    - model: Trained machine learning model.
    - X_train: DataFrame, training features.
    - X_test: DataFrame, testing features.
    - y_train: Series, actual training target values.
    - y_test: Series, actual testing target values.

    Returns:
    - train_results: DataFrame, contains features, actual and predicted values of the training set.
    - test_results: DataFrame, contains features, actual and predicted values of the testing set.
    """
    # Predicting the targets
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Merging the results with the original data
    train_results = X_train.copy()
    train_results['Actual'] = y_train
    train_results['Predicted'] = y_pred_train

    test_results = X_test.copy()
    test_results['Actual'] = y_test
    test_results['Predicted'] = y_pred_test

    return train_results, test_results
