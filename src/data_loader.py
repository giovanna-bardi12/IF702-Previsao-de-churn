from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(data_path = None):
    if data_path is None:
        project_root = Path(__file__).resolve().parents[1]

        data_path = (project_root/"data"/"data_raw"/"customer_churn_telecom_services.csv")

    else: 
        data_path = Path(data_path)

    dataframe = pd.read_csv(data_path)
    dataframe.columns = dataframe.columns.str.strip()

    return dataframe

def split_data(dataframe, target_column = "Churn", test_size = 0.2, seed = 42):
    x = dataframe.drop (columns = [target_column])
    y = dataframe[target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, stratify=y)

    return x_train, x_test, y_train, y_test

def data_loader(data_path):
    data = load_data(data_path)
    return split_data(data)