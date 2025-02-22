import pickle as pkl
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
import pandas.api.types
import polars as pl
from lifelines.utils import concordance_index
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir: Path):
    df_train = pl.read_csv(data_dir / "train.csv")
    df_test = pl.read_csv(data_dir / "test.csv")

    df_train = add_features(df_train)
    df_test = add_features(df_test)

    primary_columns = ["ID", "efs", "efs_time"]
    feature_columns = [c for c in df_train.columns if c not in primary_columns]

    categorical_columns = []
    continuous_columns = []
    for dtype, column in zip(df_train[feature_columns].dtypes, feature_columns):
        if dtype == pl.String:
            categorical_columns.append(column)
        else:
            continuous_columns.append(column)

    column_dict = {
        "primary": primary_columns,
        "feature": feature_columns,
        "categorical": categorical_columns,
        "continuous": continuous_columns,
    }
    return df_train, df_test, column_dict


def add_features(df):
    df = add_features_1(df)
    df = add_not_done(df)
    df = add_aggregate(df)
    df = add_decompose(df)
    return df


def add_features_1(df_):
    df = df_.clone()
    # replace
    df = df.with_columns(
        pl.when(pl.col("year_hct") == 2020)
        .then(19)
        .otherwise(pl.col("year_hct") - 2000)
        .alias("year_hct"),  # 2020年は4件のみなので2019に含める
        (pl.col("age_at_hct") // 10).alias("age_at_hct"),
        (pl.col("donor_age") // 10).alias("donor_age"),
        pl.when(pl.col("karnofsky_score") <= 40)
        .then(50)
        .otherwise(pl.col("karnofsky_score"))
        .alias("karnofsky_score"),  # 40以下は10件のみなので50に含める
        pl.when(pl.col("hla_high_res_8") <= 2)
        .then(3)
        .otherwise(pl.col("hla_high_res_8"))
        .alias("hla_high_res_8"),  # 以下同様
        pl.when(pl.col("hla_high_res_6") <= 1)
        .then(2)
        .otherwise(pl.col("hla_high_res_6"))
        .alias("hla_high_res_6"),
        pl.when(pl.col("hla_high_res_10") <= 3)
        .then(4)
        .otherwise(pl.col("hla_high_res_10"))
        .alias("hla_high_res_10"),
        pl.when(pl.col("hla_low_res_8") <= 2)
        .then(3)
        .otherwise(pl.col("hla_low_res_8"))
        .alias("hla_low_res_8"),
    )

    # cross feature
    df = df.with_columns(
        (pl.col("donor_age") - pl.col("age_at_hct")).alias("donor_age-age_at_hct"),
        (pl.col("comorbidity_score") + pl.col("karnofsky_score")).alias(
            "comorbidity_score+karnofsky_score"
        ),
        (pl.col("comorbidity_score") - pl.col("karnofsky_score")).alias(
            "comorbidity_score-karnofsky_score"
        ),
        (pl.col("comorbidity_score") * pl.col("karnofsky_score")).alias(
            "comorbidity_score*karnofsky_score"
        ),
        (pl.col("comorbidity_score") / pl.col("karnofsky_score")).alias(
            "comorbidity_score/karnofsky_score"
        ),
    )
    return df


def add_not_done(df_):
    columns = [
        "psych_disturb",
        "diabetes",
        "arrhythmia",
        "renal_issue",
        "pulm_severe",
        "obesity",
        "hepatic_severe",
        "prior_tumor",
        "peptic_ulcer",
        "rheum_issue",
        "hepatic_mild",
        "cardiac",
        "pulm_moderate",
    ]
    df = df_.clone()
    df = df.with_columns(
        *[
            pl.when(pl.col(c).is_null())
            .then(pl.lit("Not done"))
            .otherwise(pl.col(c))
            .alias(c)
            for c in columns
        ]
    )
    return df


def add_aggregate(df_):
    df = df_.clone()
    # dri_score
    df = df.with_columns(
        pl.col("dri_score")
        .str.contains("TED AML case <missing cytogenetics")
        .cast(pl.String)
        .alias("dri_score_TED"),
        (
            pl.when(pl.col("dri_score") == "High - TED AML case <missing cytogenetics")
            .then(pl.lit("High"))
            .when(
                pl.col("dri_score")
                == "Intermediate - TED AML case <missing cytogenetics"
            )
            .then(pl.lit("Intermediate"))
            .when(
                pl.col("dri_score").is_in(
                    [
                        "Missing disease status",
                        "N/A - disease not classifiable",
                        "TBD cytogenetics",
                    ]
                )
                | pl.col("dri_score").is_null()
            )
            .then(pl.lit("Unknow"))
            .otherwise(pl.col("dri_score"))
        ).alias("dri_score_aggregate"),
    )
    # cyto_score, cyto_score_detail
    df = df.with_columns(
        (
            pl.when(pl.col("cyto_score").is_in(["Not tested", "TBD"]))
            .then(pl.lit("No Result"))
            .when(pl.col("cyto_score").is_in(["Favorable", "Normal"]))
            .then(pl.lit("More than Normal"))
            .otherwise(pl.col("cyto_score"))
        ).alias("cyto_score_aggregate")
    )
    df = df.with_columns(
        pl.when(pl.col("cyto_score_detail").is_in(["Not tested", "TBD"]))
        .then(pl.lit("No Result"))
        .otherwise(pl.col("cyto_score_detail"))
        .alias("cyto_score_detail_aggregate"),
        (pl.col("cyto_score") == pl.col("cyto_score_detail"))
        .cast(pl.String)
        .alias("is_cyto_score_same"),
    )
    # conditioning_intensity
    df = df.with_columns(
        (
            pl.when(
                pl.col("conditioning_intensity").is_in(
                    ["TBD", "No drugs reported", "N/A, F(pre-TED) not submitted"]
                )
            )
            .then(pl.lit("No Result"))
            .otherwise(pl.col("conditioning_intensity"))
        ).alias("conditioning_intensity_aggregate")
    )
    return df


def add_decompose(df_):
    df = df_.clone()
    # tbi_status
    df = df.with_columns(
        ((pl.col("tbi_status") != "No TBI") & (pl.col("tbi_status").str.contains("Cy")))
        .cast(pl.String)
        .alias("TBI_and_plus_Cy"),
        (
            (pl.col("tbi_status") != "No TBI")
            & (pl.col("tbi_status").str.contains("-cGy"))
        )
        .cast(pl.String)
        .alias("TBI_and_minus_cGy"),
        (
            (pl.col("tbi_status") != "No TBI")
            & (pl.col("tbi_status").str.contains(">cGy"))
        )
        .cast(pl.String)
        .alias("TBI_and_greater_cGy"),
        (
            (pl.col("tbi_status") != "No TBI")
            & (pl.col("tbi_status").str.contains("<=cGy"))
        )
        .cast(pl.String)
        .alias("TBI_and_less_cGy"),
        (pl.col("tbi_status") == "No TBI")
        .cast(pl.String)
        .alias("tbi_status_decompose"),
    )
    # cmv_status
    df = df.with_columns(
        (
            pl.col("cmv_status").is_not_null()
            & (pl.col("cmv_status").str.starts_with("+/"))
        )
        .cast(pl.String)
        .alias("cmv_status_from_plus"),
        (
            pl.col("cmv_status").is_not_null()
            & (pl.col("cmv_status").str.ends_with("/+"))
        )
        .cast(pl.String)
        .alias("cmv_status_to_plus"),
    )
    # tce_imm_match
    df = df.with_columns(
        (
            pl.col("tce_imm_match").is_not_null()
            & (pl.col("tce_imm_match").str.starts_with("P/"))
        )
        .cast(pl.String)
        .alias("tce_imm_match_from_P"),
        (
            pl.col("tce_imm_match").is_not_null()
            & (pl.col("tce_imm_match").str.starts_with("H/"))
        )
        .cast(pl.String)
        .alias("tce_imm_match_from_H"),
        (
            pl.col("tce_imm_match").is_not_null()
            & (pl.col("tce_imm_match").str.starts_with("G/"))
        )
        .cast(pl.String)
        .alias("tce_imm_match_from_G"),
        (
            pl.col("tce_imm_match").is_not_null()
            & (pl.col("tce_imm_match").str.ends_with("/B"))
        )
        .cast(pl.String)
        .alias("tce_imm_match_to_B"),
        (
            pl.col("tce_imm_match").is_not_null()
            & (pl.col("tce_imm_match").str.ends_with("/P"))
        )
        .cast(pl.String)
        .alias("tce_imm_match_to_P"),
        (
            pl.col("tce_imm_match").is_not_null()
            & (pl.col("tce_imm_match").str.ends_with("/H"))
        )
        .cast(pl.String)
        .alias("tce_imm_match_to_H"),
        (
            pl.col("tce_imm_match").is_not_null()
            & (pl.col("tce_imm_match").str.ends_with("/G"))
        )
        .cast(pl.String)
        .alias("tce_imm_match_to_G"),
    )
    # tce_match
    df = df.with_columns(
        pl.when(pl.col("tce_match").is_not_null())
        .then(pl.lit("Null"))
        .when(pl.col("tce_match").is_in(["HvG non-permissive", "GvH non-permissive"]))
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("tce_match_non_permissive"),
        pl.when(pl.col("tce_match").is_not_null())
        .then(pl.lit("Null"))
        .when(pl.col("tce_match").is_in(["Fully matched", "Permissive"]))
        .then(pl.lit("Yes"))
        .otherwise(pl.lit("No"))
        .alias("tce_match_permissive"),
    )
    # sex_match
    df = df.with_columns(
        (
            pl.col("sex_match").is_not_null()
            & (pl.col("sex_match").str.starts_with("M-"))
        )
        .cast(pl.String)
        .alias("sex_match_from_M"),
        (pl.col("sex_match").is_not_null() & (pl.col("sex_match").str.ends_with("-M")))
        .cast(pl.String)
        .alias("sex_match_to_M"),
        (
            pl.col("sex_match").is_not_null()
            & (pl.col("sex_match").is_in(["M-M", "F-F"]))
        )
        .cast(pl.String)
        .alias("sex_match_full"),
    )
    # tce_div_match
    df = df.with_columns(
        (
            pl.col("tce_div_match").is_not_null()
            & pl.col("tce_div_match").str.contains("non-permissive")
        )
        .cast(pl.String)
        .alias("tce_div_match_non_permissive")
    )
    return df


def preprocess(
    df,
    column_dict,
    categorical_transform_dict=None,
):
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

    return df, column_dict, categorical_transform_dict


def get_fold(df_train_preprocess, df_train, num_fold):
    kf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=43)
    kf_group = (
        df_train["efs"].cast(pl.String)
        + df_train["race_group"].cast(pl.String)
        + (df_train["age_at_hct"] == 0.044).cast(pl.String)
    )

    folds = np.zeros(len(df_train_preprocess))
    for i, (_, test_index) in enumerate(kf.split(df_train_preprocess, kf_group)):
        folds[test_index] = i
    df_train_preprocess = df_train_preprocess.with_columns(
        pl.Series(folds).alias("fold")
    )
    return df_train_preprocess


def racewise_pair(df, race_group):
    df = df.with_columns(pl.Series(np.arange(len(df))).alias("index"))

    pair = []
    for race in race_group:
        df_race = df.filter(pl.col("race_group") == race)

        df_left = (
            df_race.filter(pl.col("efs") == 1)
            .select(["index", "efs_time", "efs"])
            .rename(
                {"index": "index_left", "efs_time": "efs_time_left", "efs": "efs_left"}
            )
        )
        df_right = df_race.select(["index", "efs_time", "efs"]).rename(
            {"index": "index_right", "efs_time": "efs_time_right", "efs": "efs_right"}
        )
        df_pair = df_left.join(df_right, how="cross")
        df_pair = df_pair.filter(pl.col("index_left") != pl.col("index_right"))
        df_pair = df_pair.with_columns(
            (
                (pl.col("efs_time_left") < pl.col("efs_time_right"))
                | ((pl.col("efs_left") == 1) & (pl.col("efs_right") == 0))
            ).alias("left_win"),
        )
        df_pair = df_pair.with_columns(
            pl.when(pl.col("left_win"))
            .then(pl.col("index_left"))
            .otherwise(pl.col("index_right"))
            .alias("index_win"),
            pl.when(~pl.col("left_win"))
            .then(pl.col("index_left"))
            .otherwise(pl.col("index_right"))
            .alias("index_lose"),
        )
        df_pair = df_pair.with_columns(pl.lit(1).alias("weight"))
        pair.append(df_pair.select(["index_win", "index_lose"]).to_numpy())
    return np.vstack(pair)


def train_catboost(
    params: dict,
    df_train: pl.DataFrame,
    df_valid: pl.DataFrame,
    feature_names: list[str],
    categorical_feature: list[str],
    save_dir: Path,
) -> catboost.CatBoostClassifier:
    save_dir.mkdir(exist_ok=True, parents=True)

    race_group = df_train["race_group"].unique().sort().to_list()

    pair_train = racewise_pair(df_train, race_group)
    pair_valid = racewise_pair(df_valid, race_group)

    query_ids_train = df_train["race_group"].to_numpy()
    query_ids_valid = df_valid["race_group"].to_numpy()

    print(f"Race group = {race_group}")
    print(f"Num train = {len(df_train)}")
    print(f"Num valid = {len(df_valid)}")
    print(f"Num train pair = {len(pair_train)}")
    print(f"Num valid pair = {len(pair_valid)}")

    df_train = df_train.to_pandas()
    df_valid = df_valid.to_pandas()

    # for col in categorical_feature:
    #     df_train[col] = df_train[col].astype("category")
    #     df_valid[col] = df_valid[col].astype("category")

    pool_train = catboost.Pool(
        data=df_train[feature_names],
        # label=df_valid["efs"],
        # cat_features=categorical_feature,
        group_id=query_ids_train,
        pairs=pair_train,
    )
    pool_valid = catboost.Pool(
        data=df_valid[feature_names],
        # label=df_valid["efs"],
        # cat_features=categorical_feature,
        group_id=query_ids_valid,
        pairs=pair_valid,
    )

    model = catboost.CatBoostRanker(**params)
    model.fit(
        pool_train,
        eval_set=pool_valid,
    )

    model.save_model(save_dir / "model.cbm")
    with open(save_dir / "eval_result.pkl", "wb") as f:
        pkl.dump(model.get_evals_result(), f)

    return model


def train_model(params, df_train, feature_names, categorical_cols, save_dir, num_fold):
    header_cols = ["ID", "efs_time", "efs", "race_group"]

    df_oof_list = []
    for fold in range(num_fold):
        print("#" * 50 + f"\n### Fold = {fold + 1}\n" + "#" * 50)
        fold_dir = save_dir / f"fold_{fold}"

        df_train_fold = df_train.filter(pl.col("fold") != fold).sort(
            ["race_group", "ID"]
        )  # グループが並んでいないとエラーになる
        df_valid_fold = df_train.filter(pl.col("fold") == fold).sort(
            ["race_group", "ID"]
        )
        df_oof = df_valid_fold.select(header_cols)

        model = train_catboost(
            params=params,
            df_train=df_train_fold,
            df_valid=df_valid_fold,
            feature_names=feature_names,
            categorical_feature=categorical_cols,
            save_dir=fold_dir,
        )

        pred_oof = inference_model(
            model=model,
            df=df_valid_fold,
            feature_names=feature_names,
            categorical_cols=categorical_cols,
        )

        df_oof = df_oof.with_columns(pl.Series(pred_oof).alias("prediction"))
        df_oof_list.append(df_oof)

    df_oof = pl.concat(df_oof_list).sort("ID")
    df_oof.write_parquet(save_dir / "df_oof.parquet")

    metric_score = validate_target(df_oof, "prediction")
    print(f"score = {metric_score}")


def inference_model(model, df, feature_names, categorical_cols):
    df = df.to_pandas()
    for col in categorical_cols:
        df[col] = df[col].astype("category")
    pred_proba = model.predict(df[feature_names])
    return pred_proba


def validate_target(df, col):
    df_true = df.select(["ID", "efs", "efs_time", "race_group"]).to_pandas()
    df_pred = df.select(["ID", col]).rename({col: "prediction"}).to_pandas()
    check_score = score(df_true, df_pred, "ID")
    return check_score


def print_score(df, col):
    race_group = df["race_group"].unique().sort().to_list()
    race_cindex_list = []
    for race in race_group:
        df_race = df.filter(pl.col("race_group") == race)
        race_cindex = validate_target(df_race, col)
        race_cindex_list.append(race_cindex)
        print(f"{race:<50} = {race_cindex}")

    metric = "metric"
    score = validate_target(df, col)
    print(f"{metric:<50} = {score}")


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str
) -> float:
    """
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> y_pred = {'prediction': {0: 1.0, 1: 0.0, 2: 1.0}}
    >>> y_pred = pd.DataFrame(y_pred)
    >>> y_pred.insert(0, row_id_column_name, range(len(y_pred)))
    >>> y_true = { 'efs': {0: 1.0, 1: 0.0, 2: 0.0}, 'efs_time': {0: 25.1234,1: 250.1234,2: 2500.1234}, 'race_group': {0: 'race_group_1', 1: 'race_group_1', 2: 'race_group_1'}}
    >>> y_true = pd.DataFrame(y_true)
    >>> y_true.insert(0, row_id_column_name, range(len(y_true)))
    >>> score(y_true.copy(), y_pred.copy(), row_id_column_name)
    0.75
    """

    del solution[row_id_column_name]
    del submission[row_id_column_name]

    event_label = "efs"
    interval_label = "efs_time"
    prediction_label = "prediction"
    for col in submission.columns:
        if not pandas.api.types.is_numeric_dtype(submission[col]):
            raise ParticipantVisibleError(f"Submission column {col} must be a number")
    # Merging solution and submission dfs on ID
    merged_df = pd.concat([solution, submission], axis=1)
    merged_df.reset_index(inplace=True)
    merged_df_race_dict = dict(merged_df.groupby(["race_group"]).groups)
    metric_list = []
    for race in merged_df_race_dict.keys():
        # Retrieving values from y_test based on index
        indices = sorted(merged_df_race_dict[race])
        merged_df_race = merged_df.iloc[indices]
        # Calculate the concordance index
        c_index_race = concordance_index(
            merged_df_race[interval_label],
            -merged_df_race[prediction_label],
            merged_df_race[event_label],
        )
        metric_list.append(c_index_race)
    return float(np.mean(metric_list) - np.sqrt(np.var(metric_list)))
