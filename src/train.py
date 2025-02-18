import pickle as pkl
from pathlib import Path

import lightgbm as lgbm
import polars as pl


def train_lgbm(
    params: dict,
    df_train: pl.DataFrame,
    df_valid: pl.DataFrame,
    feature_names: list[str],
    categorical_feature: list[str],
    save_dir: Path,
) -> lgbm.Booster:
    save_dir.mkdir(exist_ok=True, parents=True)

    print(f"Num train = {len(df_train)}")
    print(f"Num valid = {len(df_valid)}")

    ds_train = lgbm.Dataset(
        df_train[feature_names].to_pandas(),
        label=df_train["label"].to_pandas(),
        free_raw_data=False,
        categorical_feature=categorical_feature,
    )
    ds_valid = lgbm.Dataset(
        df_valid[feature_names].to_pandas(),
        label=df_valid["label"].to_pandas(),
        reference=ds_train,
        free_raw_data=False,
        categorical_feature=categorical_feature,
    )

    early_stopping_callback = lgbm.early_stopping(
        stopping_rounds=params["early_stopping_round"], first_metric_only=True
    )
    log_evaluation_callback = lgbm.log_evaluation(period=params["early_stopping_round"])
    eval_result = {}
    record_evaluation = lgbm.record_evaluation(eval_result)

    model = lgbm.train(
        params=params,
        train_set=ds_train,
        valid_sets=(ds_valid, ds_train),
        callbacks=[
            early_stopping_callback,
            log_evaluation_callback,
            record_evaluation,
        ],
    )

    model.save_model(save_dir / "model.txt")
    with open(save_dir / "eval_result.pkl", "wb") as f:
        pkl.dump(eval_result, f)

    return model
