import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def handle_missing_values(x_train, x_test, type="Cut"): #tipos: Cut, Media, Mediana, Moda
    x_train = x_train.copy()
    x_test = x_test.copy().reindex(columns=x_train.columns)

    categorical_columns = x_train.select_dtypes( include=["object", "category", "bool", "string"]).columns
    numeric_columns = x_train.select_dtypes(include="number").columns

    if type == "Cut":
        x_train= x_train.dropna() 
        x_test = x_test.dropna()

    if type in ["Media", "Mediana"]:
        for column in numeric_columns:
            if x_train[column].isna().any() or x_test[column].isna().any():
                if type == "Media":
                    value = x_train[column].mean()
                else:
                    value = x_train[column].median()

                x_train[column] = x_train[column].fillna(value)
                x_test[column] = x_test[column].fillna(value)

    if type == "Moda":
        columns_to_fill = x_train.columns
    else:
        columns_to_fill = categorical_columns

    for column in columns_to_fill:
        if x_train[column].isna().any() or x_test[column].isna().any():
            modes = x_train[column].mode(dropna=True)

            value = modes.iloc[0]
            x_train[column] = x_train[column].fillna(value)
            x_test[column] = x_test[column].fillna(value)

    return x_train, x_test


def handle_outliers(x_train, x_test, type="Clip"): #tipos: Clip, Cut
    x_train = x_train.copy()
    x_test = x_test.copy().reindex(columns=x_train.columns)
    numeric_columns = x_train.select_dtypes(include="number").columns

    if len(numeric_columns) == 0:
            return x_train, x_test

    bounds = {}
    for column in numeric_columns:
        q1 = x_train[column].quantile(0.25)
        q3 = x_train[column].quantile(0.75)
        iqr = q3 - q1
        bounds[column] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    for column, (lower, upper) in bounds.items():
        if type == "Clip":
            x_train[column] = x_train[column].clip(lower=lower, upper=upper)
            x_test[column] = x_test[column].clip(lower=lower, upper=upper)
        if type == "Cut":
            x_train = x_train[
                x_train[column].between(lower, upper) | x_train[column].isna()]
            
            x_test = x_test[
                x_test[column].between(lower, upper) | x_test[column].isna()]
            
    return x_train, x_test


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
