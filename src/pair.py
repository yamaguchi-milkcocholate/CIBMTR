from copy import deepcopy

import numpy as np
import polars as pl


def create_pair_dataset(
    df: pl.DataFrame, column_dict: dict[str, list[str]], seed: int = 43
) -> pl.DataFrame:
    df_pair = create_pair(df=df, column_dict=column_dict)
    df_pair = shuffle_pair(df=df_pair, column_dict=column_dict, seed=seed)
    df_feature, column_dict_feature = pair_feature(
        df=df, df_pair=df_pair, column_dict=column_dict
    )
    return df_feature, column_dict_feature


def create_pair(df: pl.DataFrame, column_dict: dict[str, list[str]]) -> pl.DataFrame:
    primary_column = [c for c in column_dict["primary"] if c != "race_group"]
    df_left = (
        # どちらかはイベントがある
        df.filter(pl.col("efs") == 1)
        .select(primary_column)
        .rename({c: f"{c}_left" for c in primary_column})
    )
    df_right = df.select(primary_column).rename(
        {c: f"{c}_right" for c in primary_column}
    )

    df = df_left.join(df_right, how="cross").filter(
        (pl.col("ID_left") != pl.col("ID_right"))  # 自分自身とペアにならない
        & (pl.col("efs_time_left") < pl.col("efs_time_right"))  # leftのtimeが小さい
    )
    return df


def shuffle_pair(
    df: pl.DataFrame, column_dict: dict[str, list[str]], seed: int
) -> pl.DataFrame:
    primary_column = [c for c in column_dict["primary"] if c != "race_group"]
    left_columns = [f"{c}_left" for c in primary_column]
    right_columns = [f"{c}_right" for c in primary_column]

    np.random.seed(seed)
    indice = np.random.rand(len(df)) < 0.5
    left_index = np.arange(len(df))[indice]
    right_index = np.arange(len(df))[~indice]
    df_left = df[left_index].clone()
    df_right = df[right_index].clone()

    df_left = df_left.with_columns(pl.lit(1).alias("label"))
    df_right = df_right.with_columns(
        *[
            pl.col(left).alias(right)
            for left, right in zip(left_columns, right_columns)
        ],
        *[
            pl.col(right).alias(left)
            for left, right in zip(left_columns, right_columns)
        ],
    )
    df_right = df_right.with_columns(pl.lit(0).alias("label"))

    df_shuffle = pl.concat([df_left, df_right]).sort(["ID_left", "ID_right"])
    return df_shuffle


def pair_feature(
    df: pl.DataFrame, df_pair: pl.DataFrame, column_dict: dict[str, list[str]]
):
    primary_pair_columns = deepcopy(df_pair.columns)
    df_feature = df_pair.clone()

    left_categorical_columns = [f"{c}_left" for c in column_dict["categorical"]]
    right_categorical_columns = [f"{c}_right" for c in column_dict["categorical"]]
    left_continuous_columns = [f"{c}_left" for c in column_dict["continuous"]]
    right_continuous_columns = [f"{c}_right" for c in column_dict["continuous"]]

    left_columns = ["ID_left"] + left_categorical_columns + left_continuous_columns
    right_columns = ["ID_right"] + right_categorical_columns + right_continuous_columns
    columns = ["ID"] + column_dict["categorical"] + column_dict["continuous"]

    # 特徴量のカラムにprefixをつける
    df_feature = df_feature.join(
        df.rename({c: c_left for c, c_left in zip(columns, left_columns)}).select(
            left_columns
        ),
        on="ID_left",
    )
    df_feature = df_feature.join(
        df.rename({c: c_right for c, c_right in zip(columns, right_columns)}).select(
            right_columns
        ),
        on="ID_right",
    )

    # 連続変数は差分
    df_feature = df_feature.with_columns(
        *[
            (pl.col(f"{c}_left") - pl.col(f"{c}_right")).alias(f"{c}_diff")
            for c in column_dict["continuous"]
        ]
    )
    add_continuous_columns = [f"{c}_diff" for c in column_dict["continuous"]]

    # カテゴリ変数は一致
    df_feature = df_feature.with_columns(
        *[
            (pl.col(f"{c}_left") == pl.col(f"{c}_right"))
            .alias(f"{c}_match")
            .cast(pl.Int64)
            for c in column_dict["categorical"]
        ]
    )
    add_categorical_columns = [f"{c}_match" for c in column_dict["categorical"]]

    column_dict_pair = deepcopy(column_dict)
    column_dict_pair["primary_pair"] = primary_pair_columns
    column_dict_pair["continuous"] = (
        left_continuous_columns + right_continuous_columns + add_continuous_columns
    )
    column_dict_pair["categorical"] = (
        left_categorical_columns + right_categorical_columns + add_categorical_columns
    )

    return df_feature, column_dict_pair
