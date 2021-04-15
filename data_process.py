import os
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

attr_classes = {
    "workclass": ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay',
                  'Never-worked', 'Retired'],
    "education": ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                  '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
    "marital-status": ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                       'Married-spouse-absent', 'Married-AF-spouse'],
    "occupation": ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                   'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                   'Priv-house-serv', 'Protective-serv', 'Armed-Forces', 'Retired', 'Student', 'None'],
    "relationship": ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    "race": ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    "sex": ['Female', 'Male'],
    "native-country": ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                       'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                       'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal',
                       'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                       'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador',
                       'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'],
    "label": ['<=50K', '>50K']
}


def int_to_one_hot(val: int, total: int) -> List[int]:
    res = [0] * total
    if val >= 0:
        res[val] = 1
    return res


def one_hot_to_int(arr: List[int]) -> int:
    assert sum(arr) == 1, "arr should be an one hot array"
    res = arr.index(1)
    return res


def df_to_arr(df: pd.DataFrame) -> np.ndarray:
    names = df.columns.tolist()
    rows = len(df)

    arr = []
    for i in range(rows):
        arr_row = []
        for col in names:
            if col in attr_classes:
                raw_val = df.iloc[i][col]
                if raw_val == "?":
                    val = -1
                else:
                    val = attr_classes[col].index(raw_val)
                if col == "label":
                    arr_row.append(val)
                else:
                    arr_row.extend(int_to_one_hot(val, len(attr_classes[col])))
            else:
                val = df.iloc[i][col]
                arr_row.append(val)
        arr.append(arr_row)
    res = np.array(arr, dtype=np.float32)
    return res


def load_csv(filename: str) -> pd.DataFrame:
    names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", "occupation",
             "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country",
             "label"]

    df = pd.read_csv(filename, header=0, names=names, sep=r",\s+")
    return df


def get_mean_std(df: pd.DataFrame):
    res = {}
    for col in ["age", "capital-gain", "capital-loss", "hours-per-week"]:
        mean = df[col].mean()
        std = df[col].std()
        res[col] = {"mean": mean, "std": std}
    return res


def convert_csv_to_arr(filename: str, mean_std: Dict[str, Dict[str, float]]) -> np.ndarray:
    df = load_csv(filename)
    # remove meaningless columns
    df.drop(columns=["fnlwgt", "education-num"], inplace=True)
    # fill null value
    df.loc[df.workclass == "Never-worked", ["occupation"]] = "None"
    df.loc[(df.age < 24) & (df.occupation == "?"), ["workclass", "occupation"]] = ["Never-worked", "Student"]
    df.loc[(df.age > 60) & (df.occupation == "?"), ["workclass", "occupation"]] = ["Retired", "Retired"]
    # normalize continuous columns
    for col in mean_std:
        mean = mean_std[col]["mean"]
        std = mean_std[col]["std"]

        df[col] = (df[col] - mean) / std

    arr = df_to_arr(df)
    return arr


def split_feature(arr: np.ndarray) -> Tuple[np.ndarray, ...]:
    feature, label = arr[:, :-1], arr[:, -1:]

    a_feature_size = 50

    a_feature = feature[:, :a_feature_size]
    b_feature = feature[:, a_feature_size:]

    return a_feature, b_feature, label


if __name__ == '__main__':
    if not os.path.exists("dataset"):
        os.makedirs("dataset/a", exist_ok=True)
        os.makedirs("dataset/b", exist_ok=True)
        os.makedirs("dataset/c", exist_ok=True)

    df = load_csv("adult.train.csv")
    # calculate and save mean and std
    mean_std = get_mean_std(df)

    arr = convert_csv_to_arr("adult.train.csv", mean_std)
    a, b, label = split_feature(arr)
    np.savez("dataset/a/train.npz", a)
    np.savez("dataset/b/train.npz", b)
    np.savez("dataset/c/train.npz", label)

    arr = convert_csv_to_arr("adult.test.csv", mean_std)
    a, b, label = split_feature(arr)
    np.savez("dataset/a/test.npz", a)
    np.savez("dataset/b/test.npz", b)
    np.savez("dataset/c/test.npz", label)
