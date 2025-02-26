from functools import cached_property, lru_cache
from typing import Any

import numpy as np
import polars as pl
import pytorch_lightning as ptl
import torch
import torch.nn as nn
from metric import score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


class Backbone(nn.Module):
    def __init__(
        self,
        numerical_dim: int,
        categorical_embedding_dim: int,
        categorical_cardinality: dict[str, int],
        categorical_cols: list[str],
        share_categorical_group: dict[str, list[str]],
        network_units: list[int],
        header_units: dict[str, list[int]],
        dropout: float,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        self.numerical_dim = numerical_dim
        self.categorical_embedding_dim = categorical_embedding_dim
        self.categorical_cardinality = categorical_cardinality
        self.categorical_cols = categorical_cols
        self.share_categorical_group = share_categorical_group

        self.network_units = network_units
        self.header_units = header_units
        self.dropout = dropout
        self.norm = norm

        self.__check_params()

        self.embedding_layer = self._create_embedding_layer()
        self.net = self._create_network()
        self.headers = self._create_headers()

    @cached_property
    def concat_hidden_dim(self) -> int:
        categorical_dim = len(self.categorical_cols) * self.categorical_embedding_dim
        share_categorical_dim = sum(
            [
                len(share_categorical_cols) * self.categorical_embedding_dim
                for share_categorical_cols in self.share_categorical_group.values()
            ]
        )
        age_cluster_dim = self.categorical_embedding_dim
        numerical_dim = self.numerical_dim
        return categorical_dim + share_categorical_dim + age_cluster_dim + numerical_dim

    @cached_property
    def header_input_dim(self) -> int:
        return self.network_units[-1]

    def _create_embedding_layer(self) -> nn.ModuleDict:
        layers = {}
        for cat_name, num_emb in self.categorical_cardinality.items():
            layers[cat_name] = nn.Embedding(
                num_embeddings=num_emb, embedding_dim=self.categorical_embedding_dim
            )
        return nn.ModuleDict(layers)

    def _create_network(self) -> nn.Sequential:
        units = [self.concat_hidden_dim] + self.network_units
        if self.norm == "batch":
            norm = nn.BatchNorm1d
        elif self.norm == "layer":
            norm = nn.LayerNorm

        layers = []
        for i in range(len(units) - 1):
            layers += [
                nn.Linear(units[i], units[i + 1]),
                norm(units[i + 1]),
                nn.GELU(),
                nn.Dropout(self.dropout),
            ]
        return nn.Sequential(*layers)

    def _create_headers(self) -> nn.ModuleDict:
        if self.norm == "batch":
            norm = nn.BatchNorm1d
        elif self.norm == "layer":
            norm = nn.LayerNorm

        headers = {}
        for header_name in self.header_units:
            units = [self.header_input_dim] + self.header_units[header_name]

            layers = []
            for i in range(len(units) - 1):
                layers += [
                    nn.Linear(units[i], units[i + 1]),
                    norm(units[i + 1]),
                    nn.GELU(),
                    nn.Dropout(self.dropout),
                ]
            layers += [nn.Linear(units[-1], 1)]
            headers[header_name] = nn.Sequential(*layers)

        return nn.ModuleDict(headers)

    def forward_categorical(self, X_cat: torch.Tensor) -> torch.Tensor:
        X_emb = torch.concat(
            [
                self.embedding_layer[cat_name](X_cat[:, i])
                for i, cat_name in enumerate(self.categorical_cols)
            ],
            dim=1,
        )
        return X_emb

    def forward_share_categorical(
        self, cat_name: str, X_cat: torch.Tensor
    ) -> torch.Tensor:
        X_emb = torch.concat(
            [self.embedding_layer[cat_name](X_cat[:, i]) for i in range(X_cat.size(1))],
            dim=1,
        )
        return X_emb

    def forward_age_cluster(self, X_weight: torch.Tensor) -> torch.Tensor:
        batch_size, num_cluster = X_weight.size()
        # (batch, cluster)
        X = torch.tile(torch.arange(num_cluster), (batch_size, 1)).to(X_weight.device)
        # (batch, emb)
        X_emb = torch.sum(
            [
                # (batch, ) * (batch, emb) -> (batch, emb)
                X_weight[:, i] * self.embedding_layer["age_cluster"](X[:, i])
                for i in range(num_cluster)
            ],
            dim=1,
        )
        return X_emb

    def forward_concat_cat_num(
        self,
        X_cat: torch.Tensor,
        X_share_cat: dict[str, torch.Tensor],
        X_age_weight: torch.Tensor,
        X_num: torch.Tensor,
    ) -> torch.Tensor:
        X_cat_emb = self.forward_categorical(X_cat=X_cat)
        X_share_cat_emb = [
            self.forward_share_categorical(cat_name=cat_name, X_cat=i_X_share_cat)
            for cat_name, i_X_share_cat in X_share_cat.items()
        ]
        X_age_emb = self.forward_age_cluster(X_weight=X_age_weight)
        X_emb = torch.concat([X_cat_emb, X_age_emb, X_num] + X_share_cat_emb, dim=1)
        return X_emb

    def forward(
        self,
        X_cat: torch.Tensor,
        X_share_cat: dict[str, torch.Tensor],
        X_age_weight: torch.Tensor,
        X_num: torch.Tensor,
    ) -> torch.Tensor:
        X_emb = self.forward_concat_cat_num(
            X_cat=X_cat, X_share_cat=X_share_cat, X_age_weight=X_age_weight, X_num=X_num
        )
        X_emb = self.net(X_emb)

        Y = {}
        for header_name, header in self.headers.items():
            Y[header_name] = header(X_emb)
        return Y

    def __check_params(self) -> None:
        if self.norm not in ("batch", "layer"):
            raise ValueError("norm")


class CIBMTRModel(ptl.LightningModule):
    def __init__(
        self,
        numerical_dim: int,
        categorical_embedding_dim: int,
        categorical_cardinality: dict[str, int],
        categorical_cols: list[str],
        share_categorical_group: dict[str, list[str]],
        reg_targets: list[str],
        binary_target: str,
        pair_target: str,
        target_coefs: dict[str, float],
        network_units: list[int],
        header_units: list[int],
        dropout: float,
        lr: float,
        lr_scheduler_factor: float,
        lr_scheduler_patience: int,
        weight_decay: float,
        norm: str = "batch",
        margin: int = 1.0,
        race_cat2name: dict[int, str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        header_units_replica = {
            reg_target: header_units
            for reg_target in reg_targets + [binary_target, pair_target]
        }
        self.backbone = Backbone(
            numerical_dim=numerical_dim,
            categorical_embedding_dim=categorical_embedding_dim,
            categorical_cardinality=categorical_cardinality,
            categorical_cols=categorical_cols,
            share_categorical_group=share_categorical_group,
            network_units=network_units,
            header_units=header_units_replica,
            dropout=dropout,
            norm=norm,
        )
        self.sigmoid = nn.Sigmoid()

        self.mse_loss_func = nn.MSELoss()
        self.bce_loss_func = nn.BCELoss()
        self.pairwise_loss_func = PairwiseHingeLoss(margin=margin)

        self.tensor_stack = TensorStack()

    @cached_property
    def share_group_name(self) -> list[str]:
        return list(self.hparams.share_categorical_group.keys())

    @cached_property
    def metric_score_cols(self) -> list[str]:
        return self.hparams.reg_targes + [self.hparams.pair_target]

    def forward(
        self,
        X_cat: torch.Tensor,
        X_share_cat: dict[str, torch.Tensor],
        X_age_weight: torch.Tensor,
        X_num: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        Y = self.backbone(
            X_cat=X_cat, X_share_cat=X_share_cat, X_age_weight=X_age_weight, X_num=X_num
        )
        Y[self.hparams.binary_target] = self.sigmoid(Y[self.hparams.binary_target])
        return Y

    def training_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        total_loss, _ = self._any_step(batch=batch, prefix="train")
        return total_loss

    def _any_step(self, batch: dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        X_cat = batch["categorical"]
        X_share_cat = {k: batch[k] for k in self.share_group_name}
        X_age_weight = batch["age"]
        X_num = batch["numerical"]
        Y_reg = {k: batch[k] for k in self.hparams.reg_targets}
        Y_binary = batch[self.hparams.binary_target]
        efs_time = batch["efs_time"]
        efs = batch["efs"]

        Y_hat = self(
            X_cat=X_cat, X_share_cat=X_share_cat, X_age_weight=X_age_weight, X_num=X_num
        )
        reg_total_loss, reg_losses = self.calc_regression_loss(Y_hat=Y_hat, Y_reg=Y_reg)
        binary_loss = self.calc_binary_loss(Y_hat=Y_hat, Y_binary=Y_binary)
        pair_loss = self.calc_pair_loss(Y_hat=Y_hat, efs_time=efs_time, efs=efs)

        total_loss = reg_total_loss + binary_loss + pair_loss

        self._log(prog_bar=True, prefix=prefix, total_loss=total_loss)
        self._log(
            prog_bar=False,
            prefix=prefix,
            reg_total_loss=reg_total_loss,
            **reg_losses,
            binary_loss=binary_loss,
            pair_loss=pair_loss,
        )
        return total_loss, Y_hat

    def validation_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        efs_time = batch["efs_time"]
        efs = batch["efs"]
        race_group = batch["race_group"]

        total_loss, Y_hat = self._any_step(batch=batch, prefix="valid")

        self.tensor_stack.append(
            **Y_hat, efs_time=efs_time, efs=efs, race_group=race_group
        )
        return total_loss

    def on_validation_epoch_end(self) -> None:
        result = self.calc_metrics()
        self.tensor_stack.clear()

        self._log(prog_bar=True, prefix="valid", cindex=result["pred_mean_total"])
        self._log(
            prog_bar=False,
            prefix="valid",
            **{k: v for k, v in result if k != "pred_mean_total"},
        )

    def test_step(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        efs_time = batch["efs_time"]
        efs = batch["efs"]
        race_group = batch["race_group"]

        total_loss, Y_hat = self._any_step(batch=batch, prefix="test")

        self.tensor_stack.append(
            **Y_hat, efs_time=efs_time, efs=efs, race_group=race_group
        )
        return total_loss

    def on_test_epoch_end(self) -> None:
        result = self.calc_metrics()
        self.tensor_stack.clear()

        self._log(prog_bar=True, prefix="test", cindex=result["pred_mean_total"])
        self._log(
            prog_bar=False,
            prefix="test",
            **{k: v for k, v in result if k != "pred_mean_total"},
        )

    def calc_metrics(self) -> dict[str, float]:
        data = self.tensor_stack.get()
        df = pl.DataFrame(data)
        df = df.with_columns(pl.Series(np.arange(len(df))).alias("ID"))
        df = df.with_columns([pl.col(col).rank() for col in self.metric_score_cols])
        df = df.with_columns(
            pl.mean_horizontal(self.metric_score_cols).alias("pred_mean")
        )
        metric_score_cols = self.metric_score_cols + ["pred_mean"]

        result = {}
        for col in metric_score_cols:
            col_result = self._calc_metrics(df=df, col=col)
            result |= col_result
        return result

    def _calc_metrics(self, df: pl.DataFrame, col: str) -> dict[str, float]:
        result = {}

        result[f"{col}_total"] = validate_target(df=df, col=col)

        race_group = df["race_group"].unique().sort().to_list()
        for race in race_group:
            df_race = df.filter(pl.col("race_group") == race)

            if (self.hparams.race_cat2name is not None) and (
                race in self.hparams.race_cat2name
            ):
                race_name = self.hparams.race_cat2name[race]
            else:
                race_name = race
            result[f"{col}_{race_name}"] = validate_target(df=df_race, col=col)

        return result

    def calc_regression_loss(
        self, Y_hat: dict[str, torch.Tensor], Y_reg: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[torch.Tensor]]:
        losses = {}
        total_loss = 0
        for reg_target in self.hparams.reg_targets:
            y_hat = Y_hat[reg_target]
            y = Y_reg[reg_target]
            loss = self.mse_loss_func(y_hat, y) * self.hparams.target_coefs[reg_target]
            losses[reg_target] = loss
            total_loss += loss
        return total_loss, losses

    def calc_binary_loss(
        self, Y_hat: dict[str, torch.Tensor], Y_binary: torch.Tensor
    ) -> torch.Tensor:
        loss = (
            self.bce_loss_func(Y_hat[self.hparams.binary_target], Y_binary)
            * self.hparams.target_coefs[self.hparams.binary_target]
        )
        return loss

    def calc_pair_loss(
        self, Y_hat: dict[str, torch.Tensor], efs_time: torch.Tensor, efs: torch.Tensor
    ):
        loss = (
            self.pairwise_loss_func(
                y_hat=Y_hat[self.hparams.pair_target], efs_time=efs_time, efs=efs
            )
            * self.hparams.target_coefs[self.hparams.pair_target]
        )
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=self.hparams.lr_scheduler_factor,
            patience=self.hparams.lr_scheduler_patience,
            verbose=True,
            threshold=1e-8,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_total_loss",
        }

    def _log(self, prog_bar: bool, prefix: str, **kwargs) -> None:
        for k, v in kwargs.items():
            self.log(f"{prefix}_{k}", v, on_epoch=True, prog_bar=prog_bar, logger=True)


class PairwiseHingeLoss(nn.Module):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @lru_cache
    def combinations(self, N: int):
        ind = torch.arange(N)
        comb = torch.combinations(ind, r=2)
        return comb.to(self.device)

    def forward(
        self, y_hat: torch.Tensor, efs_time: torch.Tensor, efs: torch.Tensor
    ) -> torch.Tensor:
        batch_size = efs_time.size(0)
        y_hat = y_hat.squeeze()
        assert batch_size == y_hat.size(0)

        comb = self.combinations(batch_size)
        comb = comb[(efs[comb[:, 0]] == 1) | (efs[comb[:, 1]] == 1)]

        pred_left = y_hat[comb[:, 0]]
        pred_right = y_hat[comb[:, 1]]
        y_left = efs_time[comb[:, 0]]
        y_right = efs_time[comb[:, 1]]

        # leftが小さい場合は1,逆は-1
        y = 2 * (y_left < y_right).to(torch.float32) - 1
        loss = nn.functional.relu(-y * (pred_left - pred_right) + self.margin)

        mask = self.get_mask(comb, efs, y_left, y_right)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def get_mask(self, comb, efs, y_left, y_right):
        # left=1, right=0なのにleft>rightのものはinvalid
        cond_1_a = (efs[comb[:, 0]] == 1) & (efs[comb[:, 1]] == 0)
        cond_1_b = y_left >= y_right
        cond_1 = cond_1_a & cond_1_b

        # left=0, right=1なのにleft<rightのものはinvalid
        cond_2_a = (efs[comb[:, 0]] == 0) & (efs[comb[:, 1]] == 1)
        cond_2_b = y_left <= y_right
        cond_2 = cond_2_a & cond_2_b

        # invalidではないものに1を立てる
        invalid = cond_1 | cond_2
        mask = ~invalid

        return mask


class DataModule(ptl.LightningDataModule):
    def __init__(
        self,
        df_train: pl.DataFrame,
        df_valid: pl.DataFrame,
        df_test: pl.DataFrame,
        batch_size: int,
        column_dict: dict[str, Any],
    ) -> None:
        super().__init__()

        self.df_train = df_train
        self.df_valid = df_valid
        self.df_test = df_test
        self.batch_size = batch_size
        self.column_dict = column_dict

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def train_dataloader(self):
        dataset = CIBMTRDataset(
            df=self.df_train, column_dict=self.column_dict, device=self.device
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return loader

    def val_dataloader(self):
        dataset = CIBMTRDataset(
            df=self.df_valid, column_dict=self.column_dict, device=self.device
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return loader

    def test_dataloader(self):
        dataset = CIBMTRDataset(
            df=self.df_test, column_dict=self.column_dict, device=self.device
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        return loader


class CIBMTRDataset(Dataset):
    def __init__(
        self, df: pl.DataFrame, column_dict: dict[str, Any], device: str
    ) -> None:
        self.df = df
        self.column_dict = column_dict
        self.device = device

        self.X_num = torch.tensor(
            df.select(self.numerical_feature).to_numpy(), dtype=torch.float32
        ).to(device)
        self.X_cat = torch.tensor(
            df.select(self.categorical_feature).to_numpy(), dtype=torch.long
        ).to(device)
        self.X_share_cat = {
            share_group: torch.tensor(
                df.select(share_cols).to_numpy(), dtype=torch.long
            ).to(device)
            for share_group, share_cols in self.share_categorical_feature.items()
        }
        self.X_age = torch.tensor(
            df.select(self.age_cluster_feature).to_numpy(), dtype=torch.float32
        ).to(device)

        self.Y_reg = {}
        for reg_target in self.reg_targets:
            self.Y_reg[reg_target] = torch.tensor(
                self._fill_zero_ifnull(df=df, col=reg_target), dtype=torch.float32
            ).to(device)

        self.Y_binary = torch.tensor(
            self._fill_zero_ifnull(df=df, col=self.binary_target), dtype=torch.float32
        ).to(device)

        self.efs = torch.tensor(
            self._fill_zero_ifnull(df=df, col="efs"), dtype=torch.float32
        ).to(device)
        self.efs_time = torch.tensor(
            self._fill_zero_ifnull(df=df, col="efs_time"), dtype=torch.float32
        ).to(device)
        self.race_group = torch.tensor(
            df["race_group"].to_numpy(), dtype=torch.long
        ).to(device)

    def _fill_zero_ifnull(self, df: pl.DataFrame, col: str) -> np.ndarray:
        if col in df.columns:
            value = df[col].to_numpy()
        else:
            value = np.zeros(len(df))
        return value

    @cached_property
    def numerical_feature(self) -> list[str]:
        return self.column_dict["numerical"]

    @cached_property
    def categorical_feature(self) -> list[str]:
        return self.column_dict["categorical"] + self.column_dict["flag"]

    @cached_property
    def share_categorical_feature(self) -> dict[str, list[str]]:
        return self.column_dict["share_categorical"]

    @cached_property
    def age_cluster_feature(self) -> list[str]:
        return self.column_dict["age_cluster_weight"]

    @cached_property
    def share_group(self) -> list[str]:
        return [k for k in self.share_categorical_feature]

    @cached_property
    def reg_targets(self) -> list[str]:
        return self.column_dict["reg_target"]

    @cached_property
    def binary_target(self) -> str:
        return self.column_dict["binary_target"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, idx: int
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        batch = {
            "numerical": self.X_num[idx],
            "categorical": self.X_cat,
            "age": self.X_age,
        }
        batch = batch | {
            share_group: X_share[idx]
            for share_group, X_share in self.X_share_cat.items()
        }
        batch = (
            batch
            | {reg_target: Y_reg[idx] for reg_target, Y_reg in self.Y_reg.items()}
            | {"binary_target": self.Y_binary[idx]}
        )
        batch = batch | {
            "efs": self.efs[idx],
            "efs_time": self.efs_time[idx],
            "race_group": self.race_group[idx],
        }
        return batch


class TensorStack:
    def __init__(self):
        self.data = {}

    def append(self, **kwargs) -> None:
        for k, v in kwargs.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(self._tensor_to_numpy(v))

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy().squeeze()

    def get(self) -> dict[str, np.ndarray]:
        return {k: np.hstack(v) for k, v in self.data.items()}

    def clear(self) -> None:
        self.data = {}


def validate_target(df, col):
    df_true = df.select(["ID", "efs", "efs_time", "race_group"]).to_pandas()
    df_pred = df.select(["ID", col]).rename({col: "prediction"}).to_pandas()
    check_score = score(df_true, df_pred, "ID")
    return check_score
