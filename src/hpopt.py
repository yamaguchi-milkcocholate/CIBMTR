import warnings
from pathlib import Path

import click
import polars as pl

from src.pair import create_pair_dataset
from src.preprocess import get_fold, label_encode, load_data
from src.train import HyperParameterTuner

warnings.simplefilter("ignore")
RACE_GROUP = "American Indian or Alaska Native"


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

    df_race = df_preprocess.filter(pl.col("race_group") == RACE_GROUP)
    df_race = get_fold(df=df_race, num_fold=3, seed=43)

    df_train = df_race.filter(pl.col("fold") != 0)
    df_valid = df_race.filter(pl.col("fold") == 0)

    df_train_dataset, column_dict_fold = create_pair_dataset(
        df=df_train, column_dict=column_dict, seed=43
    )
    df_valid_dataset, _ = create_pair_dataset(
        df=df_valid, column_dict=column_dict, seed=43
    )

    feature_names = column_dict_fold["continuous"] + column_dict_fold["categorical"]

    hp_tuner = HyperParameterTuner(
        df_train=df_train_dataset,
        df_valid=df_valid_dataset,
        feature_names=feature_names,
        categorical_feature=column_dict_fold["categorical"],
    )
    df_result = hp_tuner.optimize(n_trials=100)
    df_result.write_parquet(out_dir / "df_hpopt.parquet")


if __name__ == "__main__":
    main()
