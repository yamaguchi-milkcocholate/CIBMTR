import pickle as pkl
import shutil
import warnings
from dataclasses import asdict, dataclass, field
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Any, Optional

import lifelines
import numpy as np
import pandas as pd
import polars as pl
import pytorch_lightning as ptl
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from matplotlib import pyplot as plt
from metric import score
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger
from pytorch_tabular.models.common.layers import ODST
from scipy.stats import norm
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.nn.model import CIBMTRDataset, CIBMTRModel, DataModule


@dataclass
class Config:
    num_age_cluster: int = 5
    num_fold: int = 10

    categorical_embedding_dim: int = 16
    network_units: list[int] = field(default_factory=lambda: [128, 64, 32])
    header_units: list[int] = field(default_factory=lambda: [32])

    batch_size: int = 2048
    epoch: int = 60
    lr: float = 0.001
    lr_scheduler_factor = 0.1
    lr_scheduler_patience = 3
    dropout: float = 0.1
    norm: str = "batch"
    weight_decay: float = 0.00001

    margin: float = 1
    target_coefs: dict[str, float] = field(
        default_factory=lambda: {
            "kaplanmeier_target": 1,
            "nelsonaalen_target": 1,
            "breslow_target": 1,
            "exponential_target": 1,
            "gamma_target": 1,
            "binary_target": 1,
            "pair_target": 1,
        }
    )

    seed: int = 43

    def to_yaml(self, file_path):
        with open(file_path, "w") as f:
            yaml.safe_dump(asdict(self), f)

    @classmethod
    def from_yaml(cls, file_path):
        with open(file_path, "r") as f:
            d = yaml.safe_load(f)
        return cls(**d)


def train_model(
    config: Config,
    df_train: pl.DataFrame,
    column_dict: dict[str, Any],
    categorical_transform_dict: dict[str, LabelEncoder],
    out_dir: Path,
) -> None:
    ptl.seed_everything(config.seed)
    p = out_dir / "logs"
    if p.exists():
        shutil.rmtree(p)

    df_oof = []
    for fold in range(config.num_fold):
        fold_dir = out_dir / f"fold_{fold}"
        fold_dir.mkdir(exist_ok=True, parents=True)

        df_train_fold = df_train.filter(pl.col("fold") != fold).clone()
        df_valid_fold = df_train.filter(pl.col("fold") == fold).clone()
        data_module = DataModule(
            df_train=df_train_fold,
            df_valid=df_valid_fold,
            df_test=None,
            column_dict=column_dict,
            batch_size=config.batch_size,
        )

        _train_model(
            config=config,
            data_module=data_module,
            categorical_transform_dict=categorical_transform_dict,
            save_dir=fold_dir,
        )


def _train_model(
    config: Config,
    data_module: DataModule,
    categorical_transform_dict: dict[str, LabelEncoder],
    column_dict: dict[str, Any],
) -> None:
    train_dataloader = data_module.train_dataloader()
    valid_dataloader = data_module.val_dataloader()
    dataset: CIBMTRDataset = train_dataloader.dataset

    categorical_cardinality = {
        k: len(le.classes_) for k, le in categorical_transform_dict.items()
    }
    for col in column_dict["flat"]:
        categorical_cardinality[col] = 2

    race_num2cat = {
        i: c for i, c in enumerate(categorical_transform_dict["race_group"].classes_)
    }

    model = create_model(
        config=config,
        numerical_dim=len(dataset.numerical_feature),
        categorical_cols=dataset.categorical_feature,
        categorical_cardinality=categorical_cardinality,
        share_categorical_group=dataset.share_categorical_feature,
        column_dict=column_dict,
        race_num2cat=race_num2cat,
    )

    checkpoint_callback = ptl.callbacks.ModelCheckpoint(
        monitor="valid_total_loss", save_top_k=1, mode="min"
    )
    logger = CSVLogger("logs", name="expt")
    trainer = ptl.Trainer(
        accelerator="cuda" if torch.cuda.is_available() else "cpu",
        max_epochs=config.epoch,
        callbacks=[
            checkpoint_callback,
            LearningRateMonitor(logging_interval="epoch"),
            TQDMProgressBar(),
        ],
        logger=logger,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    trainer.test(model=model, dataloaders=valid_dataloader, ckpt_path="best")


def create_model(
    config: Config,
    numerical_dim: int,
    categorical_cardinality: dict[str, int],
    categorical_cols: list[str],
    share_categorical_group: dict[str, list[str]],
    column_dict: dict[str, Any],
    race_num2cat: dict[int, str],
    model_file_path: Optional[Path] = None,
    hparams_file_path: Optional[Path] = None,
) -> CIBMTRModel:
    if (model_file_path is not None) and (hparams_file_path is not None):
        with open(hparams_file_path, "r") as f:
            hparams = yaml.safe_load(f)

        model = CIBMTRModel.load_from_checkpoint(
            checkpoint_path=model_file_path, **hparams
        )
    else:
        model = CIBMTRModel(
            numerical_dim=numerical_dim,
            categorical_embedding_dim=config.categorical_embedding_dim,
            categorical_cardinality=categorical_cardinality,
            categorical_cols=categorical_cols,
            share_categorical_group=share_categorical_group,
            reg_targets=column_dict["reg_target"],
            binary_target=column_dict["binary_target"],
            pair_target=column_dict["pair_target"],
            target_coefs=config.target_coefs,
            network_units=config.network_units,
            header_units=config.header_units,
            dropout=config.dropout,
            lr=config.lr,
            lr_scheduler_factor=config.lr_scheduler_factor,
            lr_scheduler_patience=config.lr_scheduler_patience,
            weight_decay=config.weight_decay,
            norm=config.norm,
            margin=config.margin,
            race_cat2name=race_num2cat,
        )
    return model
