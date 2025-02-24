import pickle as pkl
from pathlib import Path

import click
import polars as pl

from src.pair import create_pair_dataset
from src.preprocess import get_fold, label_encode, load_data
from src.train import train_lgbm

RACE_GROUP = [
    "American Indian or Alaska Native",
    "Asian",
    "Black or African-American",
    "More than one race",
    "Native Hawaiian or other Pacific Islander",
    "White",
]
NUM_FOLD = 5


@click.command()
@click.option("--debug/--no-debug", type=bool, is_flag=True)
def main(debug: bool) -> None:
    root_dir = Path(__file__).resolve().parent.parent
    data_dir = root_dir / "data"
    out_dir = root_dir / "out"
    out_dir.mkdir(exist_ok=True, parents=True)

    df, _, column_dict = load_data(data_dir=data_dir)
    print(df["race_group"].unique().sort().to_list())
    if debug:
        df = df.head(200)

    df_preprocess, categorical_transform_dict = label_encode(
        df_=df, column_dict=column_dict
    )
    with open(out_dir / "categorical_transform_dict.pkl", "wb") as f:
        pkl.dump(categorical_transform_dict, f)

    params = {
        "objective": "binary",
        "metric": "auc",
        "num_leaves": 5,
        "feature_fraction": 0.09724083464897226,
        "bagging_fraction": 0.037088945757790925,
        "bagging_freq": 6,
        "lambda_l1": 3.4021288169850685e-05,
        "lambda_l2": 3.5995838920722445,
        "num_iterations": 10000,
        "verbosity": -1,
        "early_stopping_round": 50,
    }

    for race in RACE_GROUP:
        race_dir = out_dir / race
        race_dir.mkdir(exist_ok=True, parents=True)

        df_race = df_preprocess.filter(pl.col("race_group") == race)
        df_race = get_fold(df=df_race, num_fold=NUM_FOLD, seed=43)
        df_race.write_parquet(race_dir / "df_race.parquet")

        for fold in range(NUM_FOLD):
            print("#" * 50 + f"\n### Fold = {fold + 1}\n" + "#" * 50)
            fold_dir = race_dir / f"fold_{fold}"

            df_train = df_race.filter(pl.col("fold") != fold)
            df_valid = df_race.filter(pl.col("fold") == fold)

            df_train_dataset, column_dict_fold = create_pair_dataset(
                df=df_train, column_dict=column_dict, seed=43 + fold
            )
            df_valid_dataset, _ = create_pair_dataset(
                df=df_valid, column_dict=column_dict, seed=43 + fold
            )

            feature_names = (
                column_dict_fold["continuous"] + column_dict_fold["categorical"]
            )

            _ = train_lgbm(
                params=params,
                df_train=df_train_dataset,
                df_valid=df_valid_dataset,
                save_dir=fold_dir,
                feature_names=feature_names,
                categorical_feature=column_dict_fold["categorical"],
            )
            with open(fold_dir / "column_dict.pkl", "wb") as f:
                pkl.dump(column_dict_fold, f)


if __name__ == "__main__":
    main()
