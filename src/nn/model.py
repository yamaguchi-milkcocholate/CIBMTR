import pickle as pkl
import warnings
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

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
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class Bockbone(nn.Module):
    def __init__(
        self,
        numerical_dim: int,
        categorical_embedding_dim: int,
        categorical_cardinality: dict[str, int],
        categorical_cols: list[str],
        share_categorical_groups: dict[str, list[str]],
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
        self.share_categorical_groups = share_categorical_groups

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
        categorical_dim = self.categorical_cols * self.categorical_embedding_dim
        share_categorical_dim = sum(
            [
                len(share_categorical_cols) * self.categorical_embedding_dim
                for share_categorical_cols in self.share_categorical_groups.values()
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
