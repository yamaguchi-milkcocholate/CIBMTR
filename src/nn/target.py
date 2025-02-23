import lifelines
import polars as pl
from metric import score


def create_target(df):
    folds = df["fold"].unique().sort().to_list()
    print(f"num fold = {len(folds)}")

    univariate_models = {
        "kaplanmeier_target": kaplanmeier_fitter,
        "nelsonaalen_target": nelsonaalen_fitter,
        "breslow_target": breslow_fitter,
        "exponential_target": exponential_fitter,
        "gamma_target": gamma_fitter,
    }
    target_cols = list(univariate_models.keys())

    df_oof_list = []
    for fold in folds:
        df_train = df.filter(pl.col("fold") != fold)
        df_valid = df.filter(pl.col("fold") == fold)
        df_oof = df_valid.select(["ID"])

        for target_col, target_func in univariate_models.items():
            target_values = target_func(df_train, df_valid)
            df_oof = df_oof.with_columns(pl.Series(target_values).alias(target_col))
        df_oof_list.append(df_oof)

    df_oof = pl.concat(df_oof_list)
    df = df.join(df_oof, on="ID").sort("ID")

    for target_col in target_cols:
        check_score = validate_target(df, target_col)
        print(f"Check {target_col} Target Score = {check_score}")

    return df, target_cols


def kaplanmeier_fitter(df_train, df_valid):
    kmf = lifelines.KaplanMeierFitter()
    kmf.fit(
        durations=df_train["efs_time"].to_numpy(),
        event_observed=df_train["efs"].to_numpy(),
    )
    target_values = kmf.survival_function_at_times(df_valid["efs_time"]).to_numpy()
    return target_values


def breslow_fitter(df_train, df_valid):
    bfh = lifelines.BreslowFlemingHarringtonFitter()
    bfh.fit(
        durations=df_train["efs_time"].to_numpy(),
        event_observed=df_train["efs"].to_numpy(),
    )
    target_values = bfh.survival_function_at_times(df_valid["efs_time"]).to_numpy()
    return target_values


def nelsonaalen_fitter(df_train, df_valid):
    naf = lifelines.NelsonAalenFitter()
    naf.fit(
        durations=df_train["efs_time"].to_numpy(),
        event_observed=df_train["efs"].to_numpy(),
    )
    target_values = -naf.cumulative_hazard_at_times(df_valid["efs_time"]).to_numpy()
    return target_values


def exponential_fitter(df_train, df_valid):
    exp = lifelines.ExponentialFitter()
    exp.fit(
        durations=df_train["efs_time"].to_numpy(),
        event_observed=df_train["efs"].to_numpy(),
    )
    target_values = exp.survival_function_at_times(df_valid["efs_time"]).to_numpy()
    return target_values


def gamma_fitter(df_train, df_valid):
    gamma = lifelines.GeneralizedGammaFitter()
    gamma.fit(
        durations=df_train["efs_time"].to_numpy(),
        event_observed=df_train["efs"].to_numpy(),
    )
    target_values = gamma.survival_function_at_times(df_valid["efs_time"]).to_numpy()
    return target_values


def validate_target(df, col):
    df_true = df.select(["ID", "efs", "efs_time", "race_group"]).to_pandas()
    df_pred = df.select(["ID", col]).rename({col: "prediction"}).to_pandas()
    check_score = score(df_true, df_pred, "ID")
    return check_score


def print_score(df, col):
    race_group = df["race_group"].unique().sort().to_list()
    for race in race_group:
        df_race = df.filter(pl.col("race_group") == race)
        race_cindex = validate_target(df_race, col)
        print(f"{race:<50} = {race_cindex}")

    metric = "metric"
    cindex = validate_target(df, col)
    print(f"{metric:<50} = {cindex}")
