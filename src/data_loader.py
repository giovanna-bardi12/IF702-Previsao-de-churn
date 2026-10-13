from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

TARGET_COLUMN = "Churn"
VALID_SIZE = 0.2
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


def data_loader(data_path, target_column=TARGET_COLUMN, valid_size=VALID_SIZE, seed=RANDOM_SEED):
    dataframe = load_data(data_path)
    x = dataframe.drop(columns=[target_column])
    y = dataframe[target_column]

    return train_test_split(
        x,
        y,
        valid_size=valid_size,
        random_state=seed,
        stratify=y,
    )
