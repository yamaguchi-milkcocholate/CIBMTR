from pathlib import Path

import numpy as np
import polars as pl
from sklearn.preprocessing import LabelEncoder


def load_data(data_dir: Path, num_age_cluster: int):
    df_train = pl.read_csv(data_dir / "train.csv")
    df_test = pl.read_csv(data_dir / "test.csv")

    df_train = add_feature(df_train)
    df_test = add_feature(df_test)

    primary_columns = ["ID", "efs", "efs_time"]
    feature_columns = [c for c in df_train.columns if c not in primary_columns]

    share_categorical_columns = {
        "hla_res": [
            "hla_low_res_6",
            "hla_low_res_8",
            "hla_low_res_10",
            "hla_high_res_6",
            "hla_high_res_8",
            "hla_high_res_10",
            "hla_nmdp_6",
        ],
        "hla_res_diff": ["hla_res_6_diff", "hla_res_8_diff", "hla_res_10_diff"],
        "hla_match": [
            "hla_match_a_high",
            "hla_match_a_low",
            "hla_match_b_high",
            "hla_match_b_low",
            "hla_match_c_high",
            "hla_match_c_low",
            "hla_match_dqb1_high",
            "hla_match_dqb1_low",
            "hla_match_drb1_high",
            "hla_match_drb1_low",
        ],
        "hla_match_diff": [
            "hla_match_a_diff",
            "hla_match_b_diff",
            "hla_match_c_diff",
            "hla_match_dqb1_diff",
            "hla_match_drb1_diff",
        ],
    }
    flatten_share_categorical_columns = sum(
        [v for v in share_categorical_columns.values()], []
    )

    cluster_columns = [
        f"age_at_hct_cluster_{i_age_cluster}"
        for i_age_cluster in range(num_age_cluster)
    ]

    categorical_columns = []
    numerical_columns = []
    num2cat_columns = []
    for dtype, column in zip(df_train[feature_columns].dtypes, feature_columns):
        num_unique = len(df_train[column].unique())
        if (dtype == pl.String) and (column not in flatten_share_categorical_columns):
            categorical_columns.append(column)
        elif (2 < num_unique) and (num_unique <= 25):
            if column not in flatten_share_categorical_columns:
                categorical_columns.append(column)
            num2cat_columns.append(column)
        else:
            # age_at_hctはクラスタ確率値を使用
            if column not in ("age_at_hct"):
                numerical_columns.append(column)

    df_train = df_train.with_columns(
        *[pl.col(col).cast(pl.Int64) for col in num2cat_columns]
    )
    df_test = df_test.with_columns(
        *[pl.col(col).cast(pl.Int64) for col in num2cat_columns]
    )

    column_dict = {
        "primary": primary_columns,
        "feature": feature_columns,
        "categorical": categorical_columns,
        "numerical": numerical_columns,
        "num2cat": num2cat_columns,
        "share_categorical": share_categorical_columns,
        "age_cluster": cluster_columns,
    }
    return df_train, df_test, column_dict


def preprocess(
    df,
    column_dict,
    num_age_cluster,
    categorical_transform_dict=None,
    missing_transform_dict=None,
    gmm=None,
):
    # age_at_hctのクラスタ
    df, gmm = get_gaussian_mixture_cluster(
        df=df, num_age_cluster=num_age_cluster, gmm=gmm
    )

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

    for share_parent_c, share_cols in column_dict["share_categorical"].items():
        if share_parent_c not in categorical_transform_dict:
            # 共有するカラムを結合してエンコード
            series_share = pl.concat([df[c] for c in share_cols])
            le = LabelEncoder()
            le.fit(series_share)
            categorical_transform_dict[share_parent_c] = le

        le = categorical_transform_dict[share_parent_c]
        for c in share_cols:
            filled_values = np.where(
                df[c].is_in(le.classes_), df[c], None
            )  # 想定されないカテゴリはNoneに置き換える
            df = df.with_columns(pl.Series(le.transform(filled_values)).alias(c))

    # donor_ageはage_at_hct_clusterの中央値で補完
    add_numerical_features = []
    if missing_transform_dict is None:
        missing_transform_dict = {
            v["age_at_hct_cluster"]: v["donor_age"]
            for v in df.group_by("age_at_hct_cluster")
            .agg(pl.col("donor_age").cast(pl.Int64).mode())
            .to_dicts()
        }
    filled_values = df["donor_age"].to_numpy()
    indicator = df["donor_age"].is_null().to_numpy().astype(float)
    age_cluster = df["age_at_hct_cluster"].to_numpy()
    for i_age_cluster in range(num_age_cluster):
        i_indice = (indicator == 1) & (age_cluster == i_age_cluster)
        filled_values[i_indice] = missing_transform_dict[i_age_cluster]

    df = df.with_columns(
        pl.Series(filled_values).alias("donor_age"),
        pl.Series(indicator).alias("donor_age_null_indicator"),
    )
    column_dict["flag"] = ["donor_age_null_indicator"]

    df = df.with_columns(
        pl.col("donor_age") / 100,
        ((pl.col("donor_age") - pl.col("age_at_hct")) / 100).alias(
            "donor_age_age_at_hct_diff"
        ),
    )
    column_dict["numerical"] = ["donor_age", "donor_age_age_at_hct_diff"]

    return (
        df,
        column_dict,
        categorical_transform_dict,
        missing_transform_dict,
        gmm,
    )


def add_feature(df):
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
    df = df.with_columns(
        pl.when(pl.col("dri_score") == "Missing disease status")
        .then(None)
        .otherwise(pl.col("dri_score"))
        .alias("dri_score"),
        pl.col("hla_low_res_6").clip(3, 10).alias("hla_low_res_6"),
        pl.col("hla_low_res_8").clip(3, 10).alias("hla_low_res_8"),
        pl.col("hla_low_res_10").clip(3, 10).alias("hla_low_res_10"),
        pl.col("hla_high_res_6").clip(3, 10).alias("hla_high_res_6"),
        pl.col("hla_high_res_8").clip(3, 10).alias("hla_high_res_8"),
        pl.col("hla_high_res_10").clip(3, 10).alias("hla_high_res_10"),
        pl.col("year_hct").clip(None, 2019).alias("year_hct"),
        pl.when(
            pl.col("gvhd_proph").is_in(
                ["FK+- others(not MMF,MTX)", "CSA +- others(not FK,MMF,MTX)"]
            )
        )
        .then(None)
        .otherwise(pl.col("gvhd_proph"))
        .alias("gvhd_proph"),
        pl.when(pl.col("tce_imm_match") == "P/G")
        .then(None)
        .otherwise(pl.col("tce_imm_match"))
        .alias("tce_imm_match"),
        (pl.col("karnofsky_score") // 10 * 10).clip(50, 100).alias("karnofsky_score"),
        pl.col("comorbidity_score")
        .cast(pl.Int64)
        .clip(0, 10)
        .alias("comorbidity_score"),
    )
    # hla_{high|low}_res_{6|8|10} diff
    levels = [6, 8, 10]
    df = df.with_columns(
        *[
            (pl.col(f"hla_high_res_{level}") - pl.col(f"hla_low_res_{level}")).alias(
                f"hla_res_{level}_diff"
            )
            for level in levels
        ]
    )
    types = ["a", "b", "c", "dqb1", "drb1"]
    df = df.with_columns(
        *[
            (pl.col(f"hla_match_{t}_high") - pl.col(f"hla_match_{t}_low")).alias(
                f"hla_match_{t}_diff"
            )
            for t in types
        ]
    )
    df = df.with_columns(
        pl.col("hla_res_6_diff").clip(-3, 3).alias("hla_res_6_diff"),
        pl.col("hla_res_8_diff").clip(-3, 3).alias("hla_res_8_diff"),
        pl.col("hla_res_10_diff").clip(-3, 3).alias("hla_res_10_diff"),
    )

    return df


def get_gaussian_mixture_cluster(
    df: pl.DataFrame, num_age_cluster: int, gmm: GaussianMixture = None
) -> tuple[pl.DataFrame, GaussianMixture]:
    X = df[["age_at_hct"]].to_numpy()
    if gmm is None:
        gmm = GaussianMixture(
            n_components=num_age_cluster, covariance_type="spherical", random_state=43
        ).fit(X)

    X_cluster_prob = gmm.predict_proba(X)
    x_cluster = gmm.predict(X)
    df = df.with_columns(
        *[
            pl.Series(X_cluster_prob[:, i]).alias(f"age_at_hct_cluster_{i}")
            for i in range(num_age_cluster)
        ],
        pl.Series(x_cluster).alias("age_at_hct_cluster"),
    )
    return df, gmm
