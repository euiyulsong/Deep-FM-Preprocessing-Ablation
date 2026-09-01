# real_correlation_experiment.py

import argparse
import random
from copy import deepcopy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss

from ucimlrepo import fetch_ucirepo


# ============================================================
# Seed
# ============================================================

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Download real dataset from UCI
# ============================================================

def load_data():

    print("Downloading UCI Online Shoppers dataset...")

    dataset = fetch_ucirepo(id=468)

    X = dataset.data.features.copy()
    y = dataset.data.targets.copy()

    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    y = y.astype(str).str.lower().map({
        "true": 1,
        "false": 0,
        "1": 1,
        "0": 0,
    })

    X["label"] = y.astype(int)

    print("Rows:", len(X))
    print("Columns:", len(X.columns) - 1)
    print("Positive rate:", X["label"].mean())

    return X


# ============================================================
# Split
# ============================================================

def split_data(df, seed=42):

    train, temp = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=seed,
    )

    val, test = train_test_split(
        temp,
        test_size=0.5,
        stratify=temp["label"],
        random_state=seed,
    )

    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


# ============================================================
# Feature definitions
# ============================================================

NUMERIC_COLS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

CATEGORICAL_COLS = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


# ============================================================
# Numerical preprocessing
# ============================================================

def preprocess_numeric(train, val, test):

    for col in NUMERIC_COLS:

        train[col] = pd.to_numeric(
            train[col],
            errors="coerce",
        ).fillna(0)

        val[col] = pd.to_numeric(
            val[col],
            errors="coerce",
        ).fillna(0)

        test[col] = pd.to_numeric(
            test[col],
            errors="coerce",
        ).fillna(0)

        mean = train[col].mean()
        std = train[col].std()

        if std < 1e-8:
            std = 1.0

        for df in [train, val, test]:
            df[col] = (
                df[col] - mean
            ) / std

    return train, val, test


# ============================================================
# Categorical encoding
# ============================================================

def preprocess_categorical(train, val, test):

    cardinalities = {}

    for col in CATEGORICAL_COLS:

        train[col] = train[col].astype(str)
        val[col] = val[col].astype(str)
        test[col] = test[col].astype(str)

        values = sorted(
            train[col].unique()
        )

        mapping = {
            value: idx + 1
            for idx, value
            in enumerate(values)
        }

        for df in [train, val, test]:
            df[col] = (
                df[col]
                .map(mapping)
                .fillna(0)
                .astype(np.int64)
            )

        cardinalities[col] = (
            len(mapping) + 1
        )

    return (
        train,
        val,
        test,
        cardinalities,
    )


# ============================================================
# Pearson correlation filtering
# ============================================================

def correlation_filter(
    train,
    numeric_cols,
    threshold,
):

    if threshold is None:
        return numeric_cols.copy(), []

    corr = (
        train[numeric_cols]
        .corr()
        .abs()
    )

    upper = corr.where(
        np.triu(
            np.ones(corr.shape),
            k=1,
        ).astype(bool)
    )

    removed = []

    for col in upper.columns:

        if (
            upper[col] > threshold
        ).any():

            removed.append(col)

    selected = [
        c
        for c in numeric_cols
        if c not in removed
    ]

    return selected, removed


# ============================================================
# Dataset
# ============================================================

class EcommerceDataset(Dataset):

    def __init__(
        self,
        df,
        numeric_cols,
    ):

        self.num = (
            df[numeric_cols]
            .values
            .astype(np.float32)
        )

        self.cat = (
            df[CATEGORICAL_COLS]
            .values
            .astype(np.int64)
        )

        self.y = (
            df["label"]
            .values
            .astype(np.float32)
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):

        return (
            torch.tensor(
                self.num[idx],
                dtype=torch.float32,
            ),
            torch.tensor(
                self.cat[idx],
                dtype=torch.long,
            ),
            torch.tensor(
                self.y[idx],
                dtype=torch.float32,
            ),
        )


# ============================================================
# Model
# ============================================================

class EcommerceMLP(nn.Module):

    def __init__(
        self,
        num_features,
        cardinalities,
        emb_dim=8,
    ):

        super().__init__()

        self.embeddings = nn.ModuleList([
            nn.Embedding(
                cardinalities[col],
                emb_dim,
            )
            for col in CATEGORICAL_COLS
        ])

        input_dim = (
            num_features
            + len(CATEGORICAL_COLS)
            * emb_dim
        )

        self.net = nn.Sequential(
            nn.Linear(
                input_dim,
                128,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(
                128,
                64,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(
                64,
                1,
            ),
        )

    def forward(
        self,
        num_x,
        cat_x,
    ):

        embeddings = []

        for i, emb in enumerate(
            self.embeddings
        ):
            embeddings.append(
                emb(cat_x[:, i])
            )

        cat_emb = torch.cat(
            embeddings,
            dim=1,
        )

        x = torch.cat(
            [num_x, cat_emb],
            dim=1,
        )

        return (
            self.net(x)
            .squeeze(-1)
        )


# ============================================================
# Evaluation
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

    for num_x, cat_x, y in loader:

        num_x = num_x.to(device)
        cat_x = cat_x.to(device)

        logits = model(
            num_x,
            cat_x,
        )

        probs = torch.sigmoid(
            logits
        )

        preds.extend(
            probs.cpu().numpy()
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
    )

    return auc, ll


# ============================================================
# Training
# ============================================================

def train_model(
    train,
    val,
    test,
    numeric_cols,
    cardinalities,
    args,
    device,
):

    train_ds = EcommerceDataset(
        train,
        numeric_cols,
    )

    val_ds = EcommerceDataset(
        val,
        numeric_cols,
    )

    test_ds = EcommerceDataset(
        test,
        numeric_cols,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size * 2,
        shuffle=False,
    )

    model = EcommerceMLP(
        len(numeric_cols),
        cardinalities,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-5,
    )

    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1
    best_state = None

    for epoch in range(
        1,
        args.epochs + 1,
    ):

        model.train()

        total_loss = 0
        total_n = 0

        for num_x, cat_x, y in train_loader:

            num_x = num_x.to(device)
            cat_x = cat_x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits = model(
                num_x,
                cat_x,
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item() * len(y)
            )

            total_n += len(y)

        val_auc, val_ll = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"epoch={epoch} "
            f"train_loss="
            f"{total_loss / total_n:.5f} "
            f"val_auc={val_auc:.5f} "
            f"val_logloss={val_ll:.5f}"
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
# Main
# ============================================================

def main(args):

    seed_everything(args.seed)

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Device:", device)

    # ------------------------------
    # Download actual real dataset
    # ------------------------------

    df = load_data()

    train, val, test = split_data(
        df,
        args.seed,
    )

    train, val, test = (
        preprocess_numeric(
            train,
            val,
            test,
        )
    )

    (
        train,
        val,
        test,
        cardinalities,
    ) = preprocess_categorical(
        train,
        val,
        test,
    )

    print()
    print(
        "=" * 100
    )
    print(
        "NUMERICAL CORRELATION MATRIX"
    )
    print(
        "=" * 100
    )

    print(
        train[
            NUMERIC_COLS
        ]
        .corr()
        .round(3)
        .to_string()
    )

    experiments = [
        ("all", None),
        ("corr_099", 0.99),
        ("corr_095", 0.95),
        ("corr_090", 0.90),
        ("corr_080", 0.80),
        ("corr_070", 0.70),
        ("corr_060", 0.60),
    ]

    results = []

    for name, threshold in experiments:

        print()
        print(
            "=" * 80
        )
        print(
            "EXPERIMENT:",
            name,
        )
        print(
            "=" * 80
        )

        selected, removed = (
            correlation_filter(
                train,
                NUMERIC_COLS,
                threshold,
            )
        )

        print(
            f"Features: "
            f"{len(NUMERIC_COLS)} "
            f"-> {len(selected)}"
        )

        print(
            "Selected:",
            selected,
        )

        print(
            "Removed:",
            removed,
        )

        seed_everything(
            args.seed
        )

        result = train_model(
            train,
            val,
            test,
            selected,
            cardinalities,
            args,
            device,
        )

        result.update({
            "variant": name,
            "threshold": threshold,
            "num_numeric": len(selected),
            "removed": len(removed),
            "removed_features":
                ",".join(removed),
        })

        results.append(
            result
        )

    results_df = pd.DataFrame(
        results
    )

    cols = [
        "variant",
        "threshold",
        "num_numeric",
        "removed",
        "test_auc",
        "test_logloss",
        "params",
        "removed_features",
    ]

    print()
    print(
        "=" * 120
    )
    print(
        "FINAL RESULTS"
    )
    print(
        "=" * 120
    )

    print(
        results_df[
            cols
        ].to_string(
            index=False
        )
    )

    results_df[
        cols
    ].to_csv(
        "real_correlation_results.csv",
        index=False,
    )

    print()
    print(
        "Saved: "
        "real_correlation_results.csv"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    main(args)
