import pickle as pkl
from pathlib import Path

import lightgbm as lgbm
import numpy as np
import optuna
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


class HyperParameterTuner:
    def __init__(
        self,
        df_train: pl.DataFrame,
        df_valid: pl.DataFrame,
        feature_names: list[str],
        categorical_feature: list[str],
    ) -> None:
        self.df_train = df_train
        self.df_valid = df_valid
        self.feature_names = feature_names
        self.categorical_feature = categorical_feature
        self.tmp_dir = Path(__file__).resolve().parent.parent / "out" / "tmp"
        self.tmp_dir.mkdir(exist_ok=True, parents=True)

    def evaluate(self, trial: optuna.Trial) -> float:

        params = {
            "objective": "binary",
            "metric": "auc",
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "min_data_in_leaf": trial.suggest_int("num_leaves", 15, 100),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "learning_rate": trial.suggest_uniform("feature_fraction", 0.01, 0.1),
            "num_iterations": 10000,
            "verbosity": -1,
            "early_stopping_round": 50,
        }

        _ = train_lgbm(
            params=params,
            df_train=self.df_train,
            df_valid=self.df_valid,
            feature_names=self.feature_names,
            categorical_feature=self.categorical_feature,
            save_dir=self.tmp_dir,
        )
        file_path = self.tmp_dir / "eval_result.pkl"
        with open(file_path, "rb") as f:
            eval_result = pkl.load(f)

        score = np.max(eval_result["valid_0"]["auc"])
        return score

    def optimize(self, n_trials: int) -> pl.DataFrame:
        study = optuna.create_study(direction="maximize")
        study.optimize(
            self.evaluate,
            n_trials=n_trials,
        )
        print(study.best_params)

        return pl.from_pandas(study.trials_dataframe())
