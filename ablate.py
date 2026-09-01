# feature_ablation_movielens.py

import os
import math
import random
import zipfile
import urllib.request
import argparse
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, log_loss
from tqdm import tqdm


# ============================================================
# Config
# ============================================================

SEED = 42


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(SEED)


# ============================================================
# Download MovieLens-1M
# ============================================================

def download_movielens(root="./data"):
    os.makedirs(root, exist_ok=True)

    target_dir = os.path.join(root, "ml-1m")

    if os.path.exists(os.path.join(target_dir, "ratings.dat")):
        print("MovieLens-1M already exists.")
        return target_dir

    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = os.path.join(root, "ml-1m.zip")

    print("Downloading MovieLens-1M...")
    urllib.request.urlretrieve(url, zip_path)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(root)

    print("Download complete.")

    return target_dir


# ============================================================
# Load raw dataset
# ============================================================

def load_raw_data(root):
    ratings_path = os.path.join(root, "ratings.dat")
    users_path = os.path.join(root, "users.dat")
    movies_path = os.path.join(root, "movies.dat")

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "item_id",
            "rating",
            "timestamp",
        ],
        encoding="latin-1",
    )

    users = pd.read_csv(
        users_path,
        sep="::",
        engine="python",
        names=[
            "user_id",
            "gender",
            "age",
            "occupation",
            "zip",
        ],
        encoding="latin-1",
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=[
            "item_id",
            "title",
            "genres",
        ],
        encoding="latin-1",
    )

    # 영화 연도 추출
    movies["movie_year"] = (
        movies["title"]
        .str.extract(r"\((\d{4})\)$")[0]
        .fillna(1995)
        .astype(int)
    )

    # 단순화를 위해 첫 번째 genre만 사용
    movies["primary_genre"] = (
        movies["genres"]
        .str.split("|")
        .str[0]
        .fillna("Unknown")
    )

    return ratings, users, movies


# ============================================================
# Positive / Negative implicit feedback dataset
# ============================================================

def build_implicit_dataset(
    ratings,
    users,
    movies,
    max_pos=200000,
    seed=42,
):
    rng = np.random.default_rng(seed)

    ratings = ratings.sort_values("timestamp").reset_index(drop=True)

    # MovieLens의 rating interaction 자체를 positive event로 간주
    if max_pos > 0 and len(ratings) > max_pos:
        idx = np.linspace(
            0,
            len(ratings) - 1,
            max_pos,
            dtype=int,
        )
        ratings = ratings.iloc[idx].copy()

    all_items = movies["item_id"].unique()

    # user가 전체 데이터에서 본 item
    user_seen = (
        ratings.groupby("user_id")["item_id"]
        .apply(set)
        .to_dict()
    )

    positive = ratings.copy()
    positive["label"] = 1

    negatives = []

    print("Negative sampling...")

    for row in tqdm(
        positive.itertuples(),
        total=len(positive),
    ):
        seen = user_seen[row.user_id]

        while True:
            neg_item = int(rng.choice(all_items))

            if neg_item not in seen:
                break

        negatives.append(
            {
                "user_id": row.user_id,
                "item_id": neg_item,
                "rating": 0,
                "timestamp": row.timestamp,
                "label": 0,
            }
        )

    negative = pd.DataFrame(negatives)

    data = pd.concat(
        [
            positive[
                [
                    "user_id",
                    "item_id",
                    "rating",
                    "timestamp",
                    "label",
                ]
            ],
            negative,
        ],
        ignore_index=True,
    )

    # User feature
    data = data.merge(
        users[
            [
                "user_id",
                "gender",
                "age",
                "occupation",
            ]
        ],
        on="user_id",
        how="left",
    )

    # Item feature
    data = data.merge(
        movies[
            [
                "item_id",
                "movie_year",
                "primary_genre",
            ]
        ],
        on="item_id",
        how="left",
    )

    # datetime
    dt = pd.to_datetime(
        data["timestamp"],
        unit="s",
    )

    data["hour"] = dt.dt.hour
    data["day_of_week"] = dt.dt.dayofweek

    # chronological split을 위해 정렬
    data = data.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return data


# ============================================================
# Chronological split
# ============================================================

def chronological_split(data):
    unique_times = np.sort(
        data["timestamp"].unique()
    )

    t80 = unique_times[
        int(len(unique_times) * 0.8)
    ]

    t90 = unique_times[
        int(len(unique_times) * 0.9)
    ]

    train = data[
        data["timestamp"] <= t80
    ].copy()

    val = data[
        (data["timestamp"] > t80)
        & (data["timestamp"] <= t90)
    ].copy()

    test = data[
        data["timestamp"] > t90
    ].copy()

    return train, val, test


# ============================================================
# Train-derived features
# ============================================================

def add_behavior_features(
    train,
    val,
    test,
):
    # positive interaction만 이용
    train_pos = train[
        train["label"] == 1
    ]

    user_activity = (
        train_pos.groupby("user_id")
        .size()
        .to_dict()
    )

    item_popularity = (
        train_pos.groupby("item_id")
        .size()
        .to_dict()
    )

    median_timestamp = train[
        "timestamp"
    ].median()

    for df in [train, val, test]:

        df["user_activity"] = (
            df["user_id"]
            .map(user_activity)
            .fillna(0)
            .astype(float)
        )

        df["item_popularity"] = (
            df["item_id"]
            .map(item_popularity)
            .fillna(0)
            .astype(float)
        )

        # 기준 시점으로부터 얼마나 시간이 흘렀는지
        df["time_from_reference_days"] = (
            df["timestamp"] - median_timestamp
        ) / 86400.0

    return train, val, test


# ============================================================
# Category encoding
# ============================================================

def encode_categories(
    train,
    val,
    test,
    columns,
):
    cardinalities = {}

    for col in columns:

        values = pd.concat(
            [
                train[col],
                val[col],
                test[col],
            ]
        ).astype(str)

        uniques = sorted(
            values.unique()
        )

        mapping = {
            v: i + 1
            for i, v in enumerate(uniques)
        }

        # 0 = unknown
        for df in [train, val, test]:
            df[col] = (
                df[col]
                .astype(str)
                .map(mapping)
                .fillna(0)
                .astype(int)
            )

        cardinalities[col] = len(
            mapping
        ) + 1

    return train, val, test, cardinalities


# ============================================================
# Numerical preprocessing
# ============================================================

def transform_numerical(
    train,
    val,
    test,
    cols,
    mode,
):
    train = train.copy()
    val = val.copy()
    test = test.copy()

    if mode == "raw":
        return (
            train,
            val,
            test,
            cols,
            [],
            {},
        )

    if mode == "minmax":

        for col in cols:

            min_v = train[col].min()
            max_v = train[col].max()

            denom = (
                max_v - min_v
            ) + 1e-8

            for df in [
                train,
                val,
                test,
            ]:
                df[col] = (
                    df[col] - min_v
                ) / denom

        return (
            train,
            val,
            test,
            cols,
            [],
            {},
        )

    if mode == "zscore":

        for col in cols:

            mean = train[col].mean()
            std = train[col].std()

            std = max(
                float(std),
                1e-8,
            )

            for df in [
                train,
                val,
                test,
            ]:
                df[col] = (
                    df[col] - mean
                ) / std

        return (
            train,
            val,
            test,
            cols,
            [],
            {},
        )

    if mode == "log_zscore":

        for col in cols:

            # negative 값이 가능한 feature 처리
            train_values = train[
                col
            ].values

            shift = max(
                0,
                -float(
                    np.min(train_values)
                ),
            )

            for df in [
                train,
                val,
                test,
            ]:
                df[col] = np.log1p(
                    np.maximum(
                        df[col] + shift,
                        0,
                    )
                )

            mean = train[
                col
            ].mean()

            std = train[
                col
            ].std()

            std = max(
                float(std),
                1e-8,
            )

            for df in [
                train,
                val,
                test,
            ]:
                df[col] = (
                    df[col] - mean
                ) / std

        return (
            train,
            val,
            test,
            cols,
            [],
            {},
        )

    if mode == "log_bucket":

        bucket_cols = []
        cardinalities = {}

        for col in cols:

            new_col = (
                col + "_bucket"
            )

            min_train = train[
                col
            ].min()

            shift = max(
                0,
                -float(min_train),
            )

            all_buckets = []

            for df in [
                train,
                val,
                test,
            ]:

                x = np.maximum(
                    df[col].values
                    + shift,
                    0,
                )

                bucket = np.floor(
                    np.log2(x + 1)
                ).astype(int)

                df[new_col] = bucket

                all_buckets.extend(
                    bucket.tolist()
                )

            max_bucket = (
                max(all_buckets)
                if all_buckets
                else 0
            )

            cardinalities[
                new_col
            ] = max_bucket + 2

            bucket_cols.append(
                new_col
            )

        return (
            train,
            val,
            test,
            [],
            bucket_cols,
            cardinalities,
        )

    raise ValueError(
        f"Unknown numerical mode: {mode}"
    )


# ============================================================
# Temporal feature encoding
# ============================================================

def add_temporal_features(
    train,
    val,
    test,
    mode,
):
    train = train.copy()
    val = val.copy()
    test = test.copy()

    num_cols = []
    cat_cols = []
    cardinalities = {}

    if mode == "raw":

        num_cols = [
            "hour",
            "day_of_week",
        ]

    elif mode == "minmax":

        for df in [
            train,
            val,
            test,
        ]:
            df["hour_norm"] = (
                df["hour"] / 23.0
            )

            df["dow_norm"] = (
                df["day_of_week"]
                / 6.0
            )

        num_cols = [
            "hour_norm",
            "dow_norm",
        ]

    elif mode == "embedding":

        cat_cols = [
            "hour",
            "day_of_week",
        ]

        cardinalities = {
            "hour": 24,
            "day_of_week": 7,
        }

    elif mode == "sincos":

        for df in [
            train,
            val,
            test,
        ]:

            df["hour_sin"] = np.sin(
                2
                * np.pi
                * df["hour"]
                / 24.0
            )

            df["hour_cos"] = np.cos(
                2
                * np.pi
                * df["hour"]
                / 24.0
            )

            df["dow_sin"] = np.sin(
                2
                * np.pi
                * df[
                    "day_of_week"
                ]
                / 7.0
            )

            df["dow_cos"] = np.cos(
                2
                * np.pi
                * df[
                    "day_of_week"
                ]
                / 7.0
            )

        num_cols = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]

    elif mode == "embedding_sincos":

        for df in [
            train,
            val,
            test,
        ]:

            df["hour_sin"] = np.sin(
                2
                * np.pi
                * df["hour"]
                / 24.0
            )

            df["hour_cos"] = np.cos(
                2
                * np.pi
                * df["hour"]
                / 24.0
            )

            df["dow_sin"] = np.sin(
                2
                * np.pi
                * df[
                    "day_of_week"
                ]
                / 7.0
            )

            df["dow_cos"] = np.cos(
                2
                * np.pi
                * df[
                    "day_of_week"
                ]
                / 7.0
            )

        cat_cols = [
            "hour",
            "day_of_week",
        ]

        cardinalities = {
            "hour": 24,
            "day_of_week": 7,
        }

        num_cols = [
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]

    else:
        raise ValueError(
            f"Unknown time mode: {mode}"
        )

    return (
        train,
        val,
        test,
        num_cols,
        cat_cols,
        cardinalities,
    )


# ============================================================
# Correlation feature removal
# ============================================================

def remove_correlated_features(
    train,
    numeric_cols,
    threshold,
):
    """
    feature-feature correlation만 사용.

    target correlation으로 feature를 제거하지 않는다.
    """

    if threshold is None:
        return numeric_cols, []

    if len(numeric_cols) <= 1:
        return numeric_cols, []

    corr = (
        train[numeric_cols]
        .corr()
        .abs()
    )

    upper = corr.where(
        np.triu(
            np.ones(
                corr.shape
            ),
            k=1,
        ).astype(bool)
    )

    remove = [
        col
        for col in upper.columns
        if any(
            upper[col]
            > threshold
        )
    ]

    selected = [
        c
        for c in numeric_cols
        if c not in remove
    ]

    return selected, remove


# ============================================================
# Dataset
# ============================================================

class RecDataset(Dataset):

    def __init__(
        self,
        df,
        cat_cols,
        num_cols,
    ):
        self.cat = (
            df[cat_cols]
            .values
            .astype(
                np.int64
            )
        )

        if len(num_cols) > 0:

            self.num = (
                df[num_cols]
                .values
                .astype(
                    np.float32
                )
            )

        else:

            self.num = np.zeros(
                (
                    len(df),
                    0,
                ),
                dtype=np.float32,
            )

        self.y = (
            df["label"]
            .values
            .astype(
                np.float32
            )
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(
        self,
        idx,
    ):
        return (
            torch.tensor(
                self.cat[idx],
                dtype=torch.long,
            ),
            torch.tensor(
                self.num[idx],
                dtype=torch.float32,
            ),
            torch.tensor(
                self.y[idx],
                dtype=torch.float32,
            ),
        )


# ============================================================
# DeepFM
# ============================================================

class DeepFM(nn.Module):

    def __init__(
        self,
        cat_cardinalities,
        num_features,
        embedding_dim=16,
        hidden_dims=(128, 64),
    ):
        super().__init__()

        self.cat_names = list(
            cat_cardinalities.keys()
        )

        self.num_features = (
            num_features
        )

        # Deep embeddings
        self.embeddings = (
            nn.ModuleList()
        )

        # FM first-order embeddings
        self.linear_embeddings = (
            nn.ModuleList()
        )

        for name in self.cat_names:

            cardinality = (
                cat_cardinalities[
                    name
                ]
            )

            self.embeddings.append(
                nn.Embedding(
                    cardinality,
                    embedding_dim,
                )
            )

            self.linear_embeddings.append(
                nn.Embedding(
                    cardinality,
                    1,
                )
            )

        # numerical first order
        if num_features > 0:

            self.num_linear = nn.Linear(
                num_features,
                1,
                bias=False,
            )

            # numerical feature를 FM latent vector로 변환
            self.num_embeddings = (
                nn.Parameter(
                    torch.randn(
                        num_features,
                        embedding_dim,
                    )
                    * 0.01
                )
            )

        else:
            self.num_linear = None
            self.num_embeddings = None

        total_fields = (
            len(
                self.cat_names
            )
            + num_features
        )

        deep_input_dim = (
            len(
                self.cat_names
            )
            * embedding_dim
            + num_features
        )

        layers = []

        prev = deep_input_dim

        for h in hidden_dims:

            layers.extend(
                [
                    nn.Linear(
                        prev,
                        h,
                    ),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                ]
            )

            prev = h

        layers.append(
            nn.Linear(
                prev,
                1,
            )
        )

        self.deep = nn.Sequential(
            *layers
        )

        self.bias = nn.Parameter(
            torch.zeros(1)
        )

    def forward(
        self,
        cat_x,
        num_x,
    ):

        # ----------------------------------
        # Categorical embeddings
        # ----------------------------------

        cat_embs = []

        linear_terms = []

        for i, emb in enumerate(
            self.embeddings
        ):

            e = emb(
                cat_x[:, i]
            )

            cat_embs.append(e)

            linear_terms.append(
                self.linear_embeddings[
                    i
                ](
                    cat_x[:, i]
                )
            )

        if len(cat_embs) > 0:

            cat_stack = torch.stack(
                cat_embs,
                dim=1,
            )

        else:

            cat_stack = None

        # ----------------------------------
        # Numerical field embeddings
        # ----------------------------------

        if self.num_features > 0:

            num_emb = (
                num_x.unsqueeze(-1)
                * self.num_embeddings.unsqueeze(
                    0
                )
            )

        else:

            num_emb = None

        # ----------------------------------
        # FM second-order
        # ----------------------------------

        fields = []

        if cat_stack is not None:
            fields.append(
                cat_stack
            )

        if num_emb is not None:
            fields.append(
                num_emb
            )

        fm_input = torch.cat(
            fields,
            dim=1,
        )

        sum_square = (
            fm_input.sum(
                dim=1
            )
            ** 2
        )

        square_sum = (
            fm_input
            ** 2
        ).sum(
            dim=1
        )

        fm_second = (
            0.5
            * (
                sum_square
                - square_sum
            )
        ).sum(
            dim=1,
            keepdim=True,
        )

        # ----------------------------------
        # First order
        # ----------------------------------

        if linear_terms:

            linear_cat = torch.stack(
                linear_terms,
                dim=1,
            ).sum(
                dim=1
            )

        else:

            linear_cat = 0

        if self.num_linear is not None:

            linear_num = (
                self.num_linear(
                    num_x
                )
            )

        else:

            linear_num = 0

        # ----------------------------------
        # Deep
        # ----------------------------------

        deep_parts = []

        if len(cat_embs) > 0:

            deep_parts.append(
                torch.cat(
                    cat_embs,
                    dim=1,
                )
            )

        if self.num_features > 0:

            deep_parts.append(
                num_x
            )

        deep_input = torch.cat(
            deep_parts,
            dim=1,
        )

        deep_out = self.deep(
            deep_input
        )

        logits = (
            linear_cat
            + linear_num
            + fm_second
            + deep_out
            + self.bias
        )

        return logits.squeeze(-1)


# ============================================================
# Train / Eval
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    preds = []
    labels = []

    for cat_x, num_x, y in loader:

        cat_x = cat_x.to(device)
        num_x = num_x.to(device)

        logits = model(
            cat_x,
            num_x,
        )

        prob = torch.sigmoid(
            logits
        )

        preds.extend(
            prob.cpu().numpy()
        )

        labels.extend(
            y.numpy()
        )

    preds = np.asarray(preds)
    labels = np.asarray(labels)

    auc = roc_auc_score(
        labels,
        preds,
    )

    ll = log_loss(
        labels,
        preds,
        labels=[0, 1],
    )

    return auc, ll


def train_model(
    train,
    val,
    test,
    cat_cols,
    num_cols,
    cardinalities,
    device,
    epochs=3,
    batch_size=2048,
):

    train_ds = RecDataset(
        train,
        cat_cols,
        num_cols,
    )

    val_ds = RecDataset(
        val,
        cat_cols,
        num_cols,
    )

    test_ds = RecDataset(
        test,
        cat_cols,
        num_cols,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=4,
    )

    model = DeepFM(
        {
            col: cardinalities[col]
            for col in cat_cols
        },
        num_features=len(
            num_cols
        ),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=1e-6,
    )

    criterion = (
        nn.BCEWithLogitsLoss()
    )

    best_auc = -1
    best_state = None

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        losses = []

        for cat_x, num_x, y in tqdm(
            train_loader,
            leave=False,
        ):

            cat_x = cat_x.to(
                device
            )

            num_x = num_x.to(
                device
            )

            y = y.to(
                device
            )

            optimizer.zero_grad()

            logits = model(
                cat_x,
                num_x,
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            losses.append(
                loss.item()
            )

        val_auc, val_ll = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"epoch={epoch} "
            f"loss={np.mean(losses):.4f} "
            f"val_auc={val_auc:.6f} "
            f"val_logloss={val_ll:.6f}"
        )

        if val_auc > best_auc:

            best_auc = val_auc

            best_state = deepcopy(
                model.state_dict()
            )

    model.load_state_dict(
        best_state
    )

    test_auc, test_ll = evaluate(
        model,
        test_loader,
        device,
    )

    params = sum(
        p.numel()
        for p in model.parameters()
    )

    return {
        "test_auc": test_auc,
        "test_logloss": test_ll,
        "params": params,
    }


# ============================================================
# One experiment
# ============================================================

def run_experiment(
    base_train,
    base_val,
    base_test,
    numerical_mode="zscore",
    temporal_mode="embedding",
    correlation_threshold=None,
    epochs=3,
    batch_size=2048,
    device="cuda",
):

    train = base_train.copy()
    val = base_val.copy()
    test = base_test.copy()

    # ------------------------------------
    # categorical base features
    # ------------------------------------

    categorical_cols = [
        "user_id",
        "item_id",
        "gender",
        "occupation",
        "primary_genre",
    ]

    (
        train,
        val,
        test,
        base_cardinalities,
    ) = encode_categories(
        train,
        val,
        test,
        categorical_cols,
    )

    # ------------------------------------
    # Base numerical features
    # ------------------------------------

    base_numeric = [
        "age",
        "movie_year",
        "user_activity",
        "item_popularity",
        "time_from_reference_days",
    ]

    (
        train,
        val,
        test,
        processed_num_cols,
        bucket_cols,
        bucket_cardinalities,
    ) = transform_numerical(
        train,
        val,
        test,
        base_numeric,
        numerical_mode,
    )

    categorical_cols += (
        bucket_cols
    )

    base_cardinalities.update(
        bucket_cardinalities
    )

    # ------------------------------------
    # Time encoding
    # ------------------------------------

    (
        train,
        val,
        test,
        time_num_cols,
        time_cat_cols,
        time_cardinalities,
    ) = add_temporal_features(
        train,
        val,
        test,
        temporal_mode,
    )

    categorical_cols += (
        time_cat_cols
    )

    base_cardinalities.update(
        time_cardinalities
    )

    numeric_cols = (
        processed_num_cols
        + time_num_cols
    )

    # ------------------------------------
    # correlation selection
    # ------------------------------------

    numeric_cols, removed = (
        remove_correlated_features(
            train,
            numeric_cols,
            correlation_threshold,
        )
    )

    print()
    print(
        "Categorical:",
        categorical_cols,
    )

    print(
        "Numerical:",
        numeric_cols,
    )

    print(
        "Removed by correlation:",
        removed,
    )

    return train_model(
        train,
        val,
        test,
        categorical_cols,
        numeric_cols,
        base_cardinalities,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
    )


# ============================================================
# Main experiments
# ============================================================

def main(args):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device,
    )

    root = download_movielens(
        args.data_dir
    )

    ratings, users, movies = (
        load_raw_data(root)
    )

    print(
        "Ratings:",
        len(ratings),
    )

    data = build_implicit_dataset(
        ratings,
        users,
        movies,
        max_pos=args.max_pos,
    )

    train, val, test = (
        chronological_split(data)
    )

    train, val, test = (
        add_behavior_features(
            train,
            val,
            test,
        )
    )

    print(
        f"train={len(train):,}"
    )
    print(
        f"val={len(val):,}"
    )
    print(
        f"test={len(test):,}"
    )

    results = []

    # ========================================================
    # Experiment 1
    # Numerical normalization
    # ========================================================

    numerical_modes = [
        "raw",
        "minmax",
        "zscore",
        "log_zscore",
        "log_bucket",
    ]

    for mode in numerical_modes:

        print()
        print(
            "=" * 80
        )

        print(
            "NUMERICAL EXPERIMENT:",
            mode,
        )

        print(
            "=" * 80
        )

        seed_everything(SEED)

        result = run_experiment(
            train,
            val,
            test,
            numerical_mode=mode,
            temporal_mode="embedding",
            correlation_threshold=None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )

        result.update(
            {
                "experiment":
                    "numerical",
                "variant":
                    mode,
            }
        )

        results.append(result)

    # ========================================================
    # Experiment 2
    # Time encoding
    # ========================================================

    temporal_modes = [
        "raw",
        "minmax",
        "embedding",
        "sincos",
        "embedding_sincos",
    ]

    for mode in temporal_modes:

        print()
        print(
            "=" * 80
        )

        print(
            "TEMPORAL EXPERIMENT:",
            mode,
        )

        print(
            "=" * 80
        )

        seed_everything(SEED)

        result = run_experiment(
            train,
            val,
            test,
            numerical_mode="zscore",
            temporal_mode=mode,
            correlation_threshold=None,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )

        result.update(
            {
                "experiment":
                    "temporal",
                "variant":
                    mode,
            }
        )

        results.append(result)

    # ========================================================
    # Experiment 3
    # Correlation filtering
    # ========================================================

    correlation_settings = [
        (
            "all",
            None,
        ),
        (
            "corr_099",
            0.99,
        ),
        (
            "corr_095",
            0.95,
        ),
        (
            "corr_090",
            0.90,
        ),
    ]

    for name, threshold in (
        correlation_settings
    ):

        print()
        print(
            "=" * 80
        )

        print(
            "CORRELATION EXPERIMENT:",
            name,
        )

        print(
            "=" * 80
        )

        seed_everything(SEED)

        result = run_experiment(
            train,
            val,
            test,
            numerical_mode="zscore",
            temporal_mode="sincos",
            correlation_threshold=threshold,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
        )

        result.update(
            {
                "experiment":
                    "correlation",
                "variant":
                    name,
            }
        )

        results.append(result)

    # ========================================================
    # Result
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_df = result_df[
        [
            "experiment",
            "variant",
            "test_auc",
            "test_logloss",
            "params",
        ]
    ]

    print()
    print(
        "=" * 100
    )

    print(
        "FINAL RESULTS"
    )

    print(
        "=" * 100
    )

    print(
        result_df.to_string(
            index=False
        )
    )

    result_df.to_csv(
        "feature_ablation_results.csv",
        index=False,
    )

    print()
    print(
        "Saved: "
        "feature_ablation_results.csv"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=2048,
    )

    parser.add_argument(
        "--max-pos",
        type=int,
        default=200000,
        help=(
            "positive interaction 수. "
            "0이면 ML-1M 전체 사용."
        ),
    )

    args = parser.parse_args()

    main(args)
