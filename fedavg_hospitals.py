"""
Federated learning across three simulated hospitals (FedAvg, from scratch).

Question: can three hospitals train one good diagnostic model together
without any of them sharing a single patient record?

Dataset: Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn.
         569 patients, 30 tumour measurements, label = malignant or benign.
         Public data only. No real hospital records are used.

We compare three approaches, all scored on the same held-out test set:
  1. Local only  - each hospital trains on its own patients and nothing else
  2. FedAvg      - hospitals train together; only model weights are shared
  3. Pooled      - all records copied to one place (the privacy-unfriendly baseline)

Run:  python fedavg_hospitals.py            (20 repeats, ~10 seconds)
      python fedavg_hospitals.py --seeds 5  (quicker)
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Hospital profiles: share of all training patients, and share of them with
# malignant tumours. A referral hospital sees more serious cases; a small
# district hospital sees few patients and very few malignant ones.
HOSPITALS = {
    "Hospital A (teaching, referral)": {"size": 0.60, "malignant": None},  # takes the remainder
    "Hospital B (regional)":           {"size": 0.30, "malignant": 0.25},
    "Hospital C (district)":           {"size": 0.10, "malignant": 0.12},
}

ROUNDS = 40        # federated rounds
LOCAL_EPOCHS = 5   # passes over local data per round
LR = 0.1           # learning rate
L2 = 1e-3          # weight decay
EPOCHS_CENTRAL = ROUNDS * LOCAL_EPOCHS  # same training budget for local and pooled models


# ---------------------------------------------------------------- data
def load_data(seed):
    data = load_breast_cancer()
    X = data.data
    y = (data.target == 0).astype(float)  # 1 = malignant (the class we must not miss)
    return train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)


def split_into_hospitals(X, y, rng):
    """Non-IID split: hospitals differ in size and in case mix."""
    n = len(y)
    mal_idx = list(rng.permutation(np.where(y == 1)[0]))
    ben_idx = list(rng.permutation(np.where(y == 0)[0]))
    shards = {}
    for name, prof in list(HOSPITALS.items())[1:]:          # B and C first
        size = int(round(prof["size"] * n))
        n_mal = int(round(prof["malignant"] * size))
        take = [mal_idx.pop() for _ in range(n_mal)] + [ben_idx.pop() for _ in range(size - n_mal)]
        shards[name] = np.array(take)
    first = list(HOSPITALS)[0]
    shards[first] = np.array(mal_idx + ben_idx)               # A gets everyone else
    shards = {k: shards[k] for k in HOSPITALS}                 # keep A, B, C order
    return {k: (X[v], y[v]) for k, v in shards.items()}


def federated_scaler(hospitals):
    """Feature means and std devs computed from per-hospital totals only
    (counts, sums, sums of squares) - no individual records leave a hospital."""
    n = sum(len(X) for X, _ in hospitals.values())
    s1 = sum(X.sum(axis=0) for X, _ in hospitals.values())
    s2 = sum((X ** 2).sum(axis=0) for X, _ in hospitals.values())
    mean = s1 / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    return mean, std


# ---------------------------------------------------------------- model
# Logistic regression written by hand so every step of FedAvg is visible.
def init_weights(d):
    return np.zeros(d + 1)  # last entry is the bias


def predict_proba(w, X):
    z = X @ w[:-1] + w[-1]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def train(w, X, y, epochs, rng, batch=16):
    """Mini-batch gradient descent on the hospital's own data."""
    w = w.copy()
    for _ in range(epochs):
        order = rng.permutation(len(y))
        for start in range(0, len(y), batch):
            b = order[start:start + batch]
            err = predict_proba(w, X[b]) - y[b]
            grad_w = X[b].T @ err / len(b) + L2 * w[:-1]
            grad_b = err.mean()
            w[:-1] -= LR * grad_w
            w[-1] -= LR * grad_b
    return w


def fedavg(hospitals, rng, X_test=None, y_test=None):
    """Federated Averaging. Only weight vectors travel between server and hospitals."""
    d = next(iter(hospitals.values()))[0].shape[1]
    global_w = init_weights(d)
    sizes = np.array([len(y) for _, y in hospitals.values()], dtype=float)
    history = []
    for _ in range(ROUNDS):
        local = [train(global_w, X, y, LOCAL_EPOCHS, rng) for X, y in hospitals.values()]
        global_w = np.average(local, axis=0, weights=sizes)   # bigger hospitals count more
        if X_test is not None:
            history.append(roc_auc_score(y_test, predict_proba(global_w, X_test)))
    return global_w, history


def score(w, X, y):
    p = predict_proba(w, X)
    pred = (p >= 0.5).astype(float)
    return {
        "accuracy": accuracy_score(y, pred),
        "sensitivity": recall_score(y, pred),  # share of malignant tumours caught
        "auc": roc_auc_score(y, p),
    }


# ---------------------------------------------------------------- experiment
def run(seed):
    rng = np.random.default_rng(seed)
    X_tr, X_te, y_tr, y_te = load_data(seed)
    hospitals = split_into_hospitals(X_tr, y_tr, rng)

    mean, std = federated_scaler(hospitals)
    hospitals = {k: ((X - mean) / std, y) for k, (X, y) in hospitals.items()}
    X_te = (X_te - mean) / std

    results = {}
    d = X_te.shape[1]
    for name, (X, y) in hospitals.items():
        w = train(init_weights(d), X, y, EPOCHS_CENTRAL, rng)
        results[f"Local only: {name}"] = score(w, X_te, y_te)

    w_fed, history = fedavg(hospitals, rng, X_te, y_te)
    results["Federated (FedAvg)"] = score(w_fed, X_te, y_te)

    X_all = np.vstack([X for X, _ in hospitals.values()])
    y_all = np.concatenate([y for _, y in hospitals.values()])
    w_pool = train(init_weights(d), X_all, y_all, EPOCHS_CENTRAL, rng)
    results["Pooled data (no privacy)"] = score(w_pool, X_te, y_te)

    profile = {k: (len(y), int(y.sum())) for k, (_, y) in hospitals.items()}
    return results, history, profile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    all_runs, histories = [], []
    for seed in range(args.seeds):
        res, hist, profile = run(seed)
        all_runs.append(res)
        histories.append(hist)
        if seed == 0:
            print("Hospital profiles (training patients, malignant cases):")
            for k, (n, m) in profile.items():
                print(f"  {k:34s} {n:4d} patients, {m:3d} malignant ({m / n:.0%})")
            print()

    methods = list(all_runs[0])
    rows = []
    print(f"Results over {args.seeds} random splits (mean ± std), same held-out test set per split")
    print(f"{'Approach':48s} {'Accuracy':>15s} {'Sensitivity':>15s} {'AUC':>15s}")
    for m in methods:
        row = {"approach": m}
        line = f"{m:48s}"
        for metric in ("accuracy", "sensitivity", "auc"):
            vals = np.array([r[m][metric] for r in all_runs])
            row[f"{metric}_mean"] = round(vals.mean(), 4)
            row[f"{metric}_std"] = round(vals.std(), 4)
            line += f" {vals.mean():8.3f} ± {vals.std():.3f}"
        rows.append(row)
        print(line)

    with open(os.path.join(args.out, "results.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Figure: sensitivity (share of malignant tumours caught) for each approach
    labels = [m.replace("Local only: ", "Alone: ").replace(" (teaching, referral)", "")
               .replace(" (regional)", "").replace(" (district)", " (district)") for m in methods]
    means = [np.mean([r[m]["sensitivity"] for r in all_runs]) * 100 for m in methods]
    stds = [np.std([r[m]["sensitivity"] for r in all_runs]) * 100 for m in methods]
    colors = ["#9E9E9E", "#9E9E9E", "#C62828", "#1F5FA8", "#2E7D32"]
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    y_pos = np.arange(len(methods))[::-1]
    for yp, mval, sval, c in zip(y_pos, means, stds, colors):
        ax.errorbar(mval, yp, xerr=sval, fmt="o", color=c, ms=8, capsize=4, lw=1.8)
        ax.text(mval, yp + 0.28, f"{mval:.1f}%", ha="center", va="bottom", fontsize=9, color=c,
                fontweight="bold")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(75, 100)
    ax.set_ylim(-0.6, len(methods) - 0.2)
    ax.set_xlabel(f"Malignant tumours correctly flagged (%), mean ± std over {len(all_runs)} splits")
    ax.set_title("Training together without sharing records catches more cancers", fontsize=10.5)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "sensitivity_by_approach.png"), dpi=160)

    # Figure 2: federated test AUC by round
    h = np.array(histories)
    rounds = np.arange(1, ROUNDS + 1)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ax.plot(rounds, h.mean(axis=0), color="#1F5FA8", lw=2, label="Federated model")
    ax.fill_between(rounds, h.mean(0) - h.std(0), h.mean(0) + h.std(0), color="#1F5FA8", alpha=0.15)
    pooled_auc = np.mean([r["Pooled data (no privacy)"]["auc"] for r in all_runs])
    ax.axhline(pooled_auc, color="#2E7D32", ls="--", label=f"Pooled data ({pooled_auc:.3f})")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Test AUC")
    ax.set_ylim(0.95, 1.0)
    ax.set_title("The federated model converges within a few rounds", fontsize=10.5)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fedavg_auc_by_round.png"), dpi=160)
    print(f"\nSaved results.csv, sensitivity_by_approach.png and fedavg_auc_by_round.png to {args.out}/")


if __name__ == "__main__":
    main()
