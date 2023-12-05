import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def preprocess(dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    X = dataset.drop('label', axis=1).values.astype(np.float32)
    y = dataset['label'].values.astype(np.int8)
    encoder = OneHotEncoder(sparse_output=False)
    y_onehot = encoder.fit_transform(y.reshape(-1, 1))
    X /= 255.0
    return X, y_onehot
