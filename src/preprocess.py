import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def handle_missing_values(x_train, x_test):
    None


def handle_outliers(x_train, x_test):
    None


def normalize_data(x_train, x_test):
    x_train = x_train.copy()
    x_test = x_test.copy().reindex(columns=x_train.columns)
    numeric_columns = x_train.select_dtypes(include="number").columns

    if len(numeric_columns) == 0:
        return x_train, x_test

    scaler = StandardScaler()
    x_train.loc[:, numeric_columns] = scaler.fit_transform(x_train[numeric_columns]) ## fit with x_train
    x_test.loc[:, numeric_columns] = scaler.transform(x_test[numeric_columns]) ## same scaler learned from x_train
    return x_train, x_test


def encode_data(x_train, x_test):
    x_train = x_train.copy()
    x_test = x_test.copy().reindex(columns=x_train.columns)
    categorical_columns = x_train.select_dtypes(
        include=["object", "category", "bool", "string"]
    ).columns

    if len(categorical_columns) == 0:
        return x_train, x_test

    encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    train_encoded = encoder.fit_transform(x_train[categorical_columns]) ## fit with x_train
    test_encoded = encoder.transform(x_test[categorical_columns]) ## same encoder learned from x_train
    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    train_encoded = pd.DataFrame(
        train_encoded, columns=encoded_columns, index=x_train.index
    )
    test_encoded = pd.DataFrame(
        test_encoded, columns=encoded_columns, index=x_test.index
    )

    x_train = x_train.drop(columns=categorical_columns).join(train_encoded)
    x_test = x_test.drop(columns=categorical_columns).join(test_encoded)
    return x_train, x_test
