import pandas as pd

from .data_loader import (data_loader)
from .preprocess import (encode_data, handle_missing_values, handle_outliers,normalize_data)

target = "Churn"


def encode_target(target):
    encoded = (
        target.astype(str)
        .str.strip()
        .str.lower()
        .map({"no": 0, "yes": 1})
    )

    return encoded.astype("int64")


def prepare_data():
    x_train, x_test, y_train, y_test = data_loader(None)

    y_train = encode_target(y_train).reset_index(drop=True)
    y_test = encode_target(y_test).reset_index(drop=True)


if __name__ == "__main__":
    prepare_data()