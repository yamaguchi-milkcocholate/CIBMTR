import pickle as pkl
from pathlib import Path

import click
import polars as pl

from src.catboost_pair_rank.utils import get_fold, load_data, preprocess, train_model

RACE_GROUP = [
    "American Indian or Alaska Native",
    "Asian",
    "Black or African-American",
    "More than one race",
    "Native Hawaiian or other Pacific Islander",
    "White",
]
NUM_FOLD = 10


@click.command()
@click.option("--debug/--no-debug", type=bool, is_flag=True)
def main(debug: bool) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    out_dir = root_dir / "out" / "catboost_pair_rank"
    out_dir.mkdir(exist_ok=True, parents=True)

    df_train, df_test, column_dict = load_data(data_dir=data_dir)
    (
        df_train_preprocess,
        column_dict_preprocess,
        categorical_transform_dict,
    ) = preprocess(df_train, column_dict)
    (
        _,
        column_dict_preprocess,
        categorical_transform_dict,
    ) = preprocess(df_test, column_dict, categorical_transform_dict)

    if debug:
        df_train = df_train.head(200)
        df_train_preprocess = df_train_preprocess.head(200)

    df_train_preprocess = get_fold(df_train_preprocess, df_train, NUM_FOLD)

    for k, v in column_dict_preprocess.items():
        print(k, len(v))

    with open(out_dir / "column_dict_preprocess.pkl", "wb") as f:
        pkl.dump(column_dict_preprocess, f)

    with open(out_dir / "categorical_transform_dict.pkl", "wb") as f:
        pkl.dump(categorical_transform_dict, f)

    ctb_params = {
        "loss_function": "PairLogitPairwise",
        "eval_metric": "PairAccuracy",
        "learning_rate": 0.1,
        "random_state": 42,
        "task_type": "CPU",
        "num_trees": 5,
        "reg_lambda": 8.0,
        "depth": 8,
        "verbose": 1,
        "early_stopping_rounds": 50,
    }

    feature_names = (
        column_dict_preprocess["categorical"] + column_dict_preprocess["continuous"]
    )

    train_model(
        params=ctb_params,
        df_train=df_train_preprocess,
        feature_names=feature_names,
        categorical_cols=column_dict_preprocess["categorical"],
        save_dir=out_dir,
        num_fold=NUM_FOLD,
    )


if __name__ == "__main__":
    main()
