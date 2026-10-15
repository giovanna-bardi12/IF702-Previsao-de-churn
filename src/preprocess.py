import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler 


def _align(x_train, x_val, x_test):
    cols = x_train.columns
    return x_train.copy(), x_val.copy().reindex(columns=cols), x_test.copy().reindex(columns=cols)



def _continuous_columns(x_train):
    """Colunas numéricas com mais de 2 valores distintos.
 
    Exclui binárias (ex.: SeniorCitizen e dummies do one-hot), que não
    precisam de normalização.
    """
    numeric = x_train.select_dtypes(include="number").columns
    return [c for c in numeric if x_train[c].nunique(dropna=True) > 2]
 


def handle_missing_values(x_train, x_val, x_test, type="Zero"): #Zero  #Monthly 
    x_train, x_val, x_test = _align(x_train, x_val, x_test)

    for column in x_train.columns:
        if column == "TotalCharges" and type in ("Zero", "Monthly"):
            # Regra de negócio: clientes novos ainda não acumularam cobranças.
            if type == "Zero":
                for x in (x_train, x_val, x_test):
                    x[column] = x[column].fillna(0.0)
            else:
                for x in (x_train, x_val, x_test):
                    x[column] = x[column].fillna(x["MonthlyCharges"])
            continue

        if pd.api.types.is_numeric_dtype(x_train[column]):
            value = x_train[column].median()
            if pd.isna(value):
                value = 0
        else:
            modes = x_train[column].mode(dropna=True)
            value = modes.iloc[0] if not modes.empty else "__missing__"

        for x in (x_train, x_val, x_test):
            x[column] = x[column].fillna(value)
 
    return x_train, x_val, x_test


def handle_outliers(x_train, x_val, x_test, type="Clip"):
    if type != "Clip":
        raise ValueError("Only type='Clip' is supported to preserve row/label alignment")

    x_train, x_val, x_test = _align(x_train, x_val, x_test)
    numeric_columns = _continuous_columns(x_train)

    for column in numeric_columns:
        q1 = x_train[column].quantile(0.25)
        q3 = x_train[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        for x in (x_train, x_val, x_test):
            x[column] = x[column].clip(lower=lower, upper=upper)

    return x_train, x_val, x_test

# não seguem normal, só aplica nas 3 contínuas
def normalize_data(x_train, x_val, x_test):
    x_train, x_val, x_test = _align(x_train, x_val, x_test)
    columns = _continuous_columns(x_train)
    if not columns:
        return x_train, x_val, x_test
 
    for x in (x_train, x_val, x_test):
        x[columns] = x[columns].astype("float64")

    scaler = MinMaxScaler()
    x_train[columns] = scaler.fit_transform(x_train[columns])
    x_val[columns] = scaler.transform(x_val[columns])
    x_test[columns] = scaler.transform(x_test[columns])
    return x_train, x_val, x_test



def encode_data(x_train, x_val, x_test):
    x_train, x_val, x_test = _align(x_train, x_val, x_test)
    categorical_columns = x_train.select_dtypes(
        include=["object"] 
    ).columns

    if len(categorical_columns) == 0:
        return x_train, x_val, x_test


    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoder.fit(x_train[categorical_columns])
    encoded_columns = encoder.get_feature_names_out(categorical_columns)

    def transform(x):
        encoded = pd.DataFrame(
            encoder.transform(x[categorical_columns]), columns=encoded_columns, index=x.index
        )
        return pd.concat([x.drop(columns=categorical_columns), encoded], axis=1)
 
    return transform(x_train), transform(x_val), transform(x_test)
