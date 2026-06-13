"""Distill the VLM judge into a small MLP "discriminator".

Trains on (CLIP image embedding -> VLM overall score) pairs accumulated in the
runs database. Once trained, scorer.py uses the resulting discriminator.pt for
near-instant scoring without loading the VLM.

Usage:
    python distill.py                  # requires >= 200 scored samples
    python distill.py --min-samples 50 --force
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

AUTOMATION_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(AUTOMATION_DIR))

import db

DEFAULT_MIN_SAMPLES = 200


def build_mlp(input_dim):
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid(),
    )


def load_dataset(conn):
    rows = conn.execute(
        "SELECT clip_emb, vlm_overall FROM runs "
        "WHERE clip_emb IS NOT NULL AND vlm_overall IS NOT NULL"
    ).fetchall()
    if not rows:
        return None, None
    X = np.stack(
        [np.frombuffer(emb, dtype=np.float32).copy() for emb, _ in rows]
    )
    y = np.array([score / 10.0 for _, score in rows], dtype=np.float32)
    return X, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-samples", type=int, default=DEFAULT_MIN_SAMPLES)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    with open(AUTOMATION_DIR / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
    output_dir = AUTOMATION_DIR / config["paths"]["output_dir"]
    conn = db.connect_db(output_dir / "runs.sqlite3")
    X, y = load_dataset(conn)
    conn.close()

    if X is None:
        print("No scored samples yet. Run runner.py + scorer.py first.")
        return
    n = len(X)
    print(f"Dataset: {n} samples, dim {X.shape[1]}")
    if n < args.min_samples and not args.force:
        print(
            f"Need at least {args.min_samples} samples (have {n}). "
            "Keep generating/scoring, or pass --force to train anyway."
        )
        return

    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    split = max(1, int(n * 0.1))
    val_idx, train_idx = idx[:split], idx[split:]

    Xt = torch.from_numpy(X[train_idx])
    yt = torch.from_numpy(y[train_idx]).unsqueeze(1)
    Xv = torch.from_numpy(X[val_idx])
    yv = torch.from_numpy(y[val_idx]).unsqueeze(1)

    model = build_mlp(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best_mae = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            val_mae = float((model(Xv) - yv).abs().mean().item()) * 10.0
        if val_mae < best_mae:
            best_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % 50 == 0:
            print(
                f"epoch {epoch + 1}: train_mse={loss.item():.4f} "
                f"val_mae={val_mae:.2f} (best {best_mae:.2f})"
            )

    out_path = output_dir / "discriminator.pt"
    torch.save(
        {
            "state_dict": best_state,
            "input_dim": X.shape[1],
            "samples": n,
            "val_mae_0_10": best_mae,
        },
        out_path,
    )
    print(f"Saved {out_path} (val MAE on 0-10 scale: {best_mae:.2f})")
    print("scorer.py will now use it automatically for new runs.")


if __name__ == "__main__":
    main()
