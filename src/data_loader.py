from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Churn"
MAJORITY_LABEL = "No"   # Classe 1 (majoritária: não churn)
MINORITY_LABEL = "Yes"  # Classe 2 (minoritária: churn)
RANDOM_SEED = 42


def load_data(data_path = None):
    if data_path is None:
        project_root = Path(__file__).resolve().parents[1]

        data_path = (project_root/"data"/"data_raw"/"customer_churn_telecom_services.csv")

    else: 
        data_path = Path(data_path)

    dataframe = pd.read_csv(data_path)
    dataframe.columns = dataframe.columns.str.strip()

    return dataframe


def _split_class(df_class, seed):
    """Divide UMA classe em 50% / 25% / 25%."""
    train, temp = train_test_split(df_class, test_size=0.50, random_state=seed)
    val, test = train_test_split(temp, test_size=0.50, random_state=seed)
    return train, val, test


def _oversample(df, target_size, seed):
    """Completa df até target_size repetindo amostras sorteadas . 
    """
    missing = target_size - len(df)
    if missing <= 0:
        return df
    extra = df.sample(n=missing, replace=True, random_state=seed)
    return pd.concat([df, extra])


def _shuffle(df, seed):
    """Aleatorização final. reset_index evita índices duplicados das repetições."""
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

 
def split_data(dataframe, target_column=TARGET_COLUMN, seed=RANDOM_SEED):
    """
    Classe 1 (majoritária) -> 50% treino | 25% validação | 25% teste
    Classe 2 (minoritária) -> 50% treino | 25% validação | 25% teste
 
    - Treino:    Classe 1 (50%) + Classe 2 (50%) repetida até igualar a Classe 1
    - Validação: Classe 1 (25%) + Classe 2 (25%) repetida até igualar a Classe 1
    - Teste:     Classe 1 (25%) + Classe 2 (25%) SEM repetição (distribuição real)
 
    """
    labels = set(dataframe[target_column].unique())
    expected = {MAJORITY_LABEL, MINORITY_LABEL}
    if labels != expected:
        raise ValueError(f"Rótulos inesperados em {target_column}: {labels} (esperado {expected})")
 
    majority = dataframe[dataframe[target_column] == MAJORITY_LABEL]
    minority = dataframe[dataframe[target_column] == MINORITY_LABEL]
 
    maj_train, maj_val, maj_test = _split_class(majority, seed)
    min_train, min_val, min_test = _split_class(minority, seed)
 
    train = pd.concat([maj_train, _oversample(min_train, len(maj_train), seed)])
    val = pd.concat([maj_val, _oversample(min_val, len(maj_val), seed)])
    test = pd.concat([maj_test, min_test])
 
    train, val, test = (_shuffle(df, seed) for df in (train, val, test))
 
    def xy(df):
        return df.drop(columns=[target_column]), df[target_column]
 
    x_train, y_train = xy(train)
    x_val, y_val = xy(val)
    x_test, y_test = xy(test)
    return x_train, x_val, x_test, y_train, y_val, y_test
 

 
def data_loader(data_path=None, target_column=TARGET_COLUMN, seed=RANDOM_SEED):
    dataframe = load_data(data_path)
    return split_data(dataframe, target_column=target_column, seed=seed)
 


def describe_split(y_train, y_val, y_test):
    for name, y in (("Treino", y_train), ("Validação", y_val), ("Teste", y_test)):
        counts = y.value_counts().to_dict()
        print(f"{name:<10} total={len(y):>5}  {counts}")



if __name__ == "__main__":
    *_, y_train, y_val, y_test = data_loader()
    describe_split(y_train, y_val, y_test)
