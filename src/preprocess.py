from pathlib import Path

import numpy as np
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir: Path):
    df_train = pl.read_csv(data_dir / "train.csv")
    df_test = pl.read_csv(data_dir / "test.csv")

    primary_columns = ["ID", "efs", "efs_time", "race_group"]
    feature_columns = [c for c in df_train.columns if c not in primary_columns]

    categorical_columns = []
    for dtype, column in zip(df_train[feature_columns].dtypes, feature_columns):
        if dtype == pl.String:
            categorical_columns.append(column)

    continuous_columns = [c for c in feature_columns if c not in categorical_columns]

    column_dict = {
        "primary": primary_columns,
        "feature": feature_columns,
        "categorical": categorical_columns,
        "continuous": continuous_columns,
    }
    return df_train, df_test, column_dict


def label_encode(
    df_: pl.DataFrame,
    column_dict: dict[str, list[str]],
    categorical_transform_dict=None,
):
    df = df_.clone()
    # カテゴリ変数はラベルエンコード
    if categorical_transform_dict is None:
        categorical_transform_dict = {}
    for c in column_dict["categorical"]:
        if c not in categorical_transform_dict:
            le = LabelEncoder()
            le.fit(df[c])
            categorical_transform_dict[c] = le

        le = categorical_transform_dict[c]
        filled_values = np.where(
            df[c].is_in(le.classes_), df[c], None
        )  # 想定されないカテゴリはNoneに置き換える
        df = df.with_columns(pl.Series(le.transform(filled_values)).alias(c))

    return df, categorical_transform_dict


def get_fold(df: pl.DataFrame, num_fold: int, seed: int = 43) -> pl.DataFrame:
    fold = np.zeros(len(df))
    kf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed)

    for i, (_, test_index) in enumerate(kf.split(df, df["efs"])):
        fold[test_index] = i
    df = df.with_columns(pl.Series(fold).alias("fold"))
    return df
