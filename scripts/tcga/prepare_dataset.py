"""
tcga_db_to_npz.py
=================
Reads the DRNet TCGA SQLite database (tcga.db) and generates a .npz file
with the semi-synthetic dose-response curves needed to compute MISE,
exactly as done in Schwab et al. (2020) "Learning Counterfactual
Representations for Estimating Individual Dose-Response Curves".

Output npz keys (compatible with tcga_optuna.py):
    feature  : [N, num_features]   RNA-seq covariates
    t        : [N]                 observed treatment type  (0 = control, 1-3 = drugs)
    d        : [N]                 observed dose            ([0, 1], 0 for control)
    y        : [N]                 factual outcome          (noisy)
    eval_y   : [N, K, M]          oracle dose-response curves (noise-free)
    eval_d   : [M]                 dose evaluation grid     ([0, 1])

Usage
-----
    python tcga_db_to_npz.py \
        --db      path/to/tcga.db \
        --min_val path/to/min_val.npy \
        --max_val path/to/max_val.npy \
        --out     path/to/tcga_curves.npz \
        --num_features 500 \
        --num_grid_points 10 \
        --seed 42
"""

import argparse
import io
import logging
import sqlite3

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("tcga_db_to_npz")


# ---------------------------------------------------------------------------
# SQLite binary array support (same as original data_access.py)
# ---------------------------------------------------------------------------
def _convert_array(blob):
    buf = io.BytesIO(blob)
    buf.seek(0)
    return np.load(buf)


sqlite3.register_converter("ARRAY", _convert_array)


# ---------------------------------------------------------------------------
# 1. Data extraction
# ---------------------------------------------------------------------------
def load_rnaseq_from_db(db_path: str, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
    """
    Returns normalised RNA-seq matrix X of shape [N, G] in [0, 1].
    Rows are ordered by DB rowid (stable ordering for reproducibility).
    """
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    rows = conn.execute(
        "SELECT data FROM rnaseq ORDER BY rowid;"
    ).fetchall()
    conn.close()

    LOG.info(f"Loaded {len(rows)} patients from DB.")
    X_raw = np.array([r[0] for r in rows], dtype=np.float32)

    denom = (max_val - min_val).astype(np.float32) + 1e-5
    X = (X_raw - min_val.astype(np.float32)) / denom
    X = np.clip(X, 0.0, 1.0)
    return X


def select_top_features(X: np.ndarray, num_features: int) -> np.ndarray:
    """Keep the top-`num_features` genes ranked by variance across patients."""
    var = np.var(X, axis=0)
    idx = np.argsort(var)[::-1][:num_features]
    LOG.info(f"Selected {num_features} features (out of {X.shape[1]}) by variance.")
    return X[:, idx].astype(np.float32)


# ---------------------------------------------------------------------------
# 2. Semi-synthetic benchmark  (faithful to Schwab et al. 2020 / DRNet)
# ---------------------------------------------------------------------------
def build_response_weights(X: np.ndarray, K: int, seed: int) -> list:
    """
    For each of the K-1 active treatments, derive a weight vector over a
    disjoint partition of features.  Weights are proportional to variance so
    that more informative genes contribute more (mirrors DRNet's feature
    selection per treatment).
    """
    rng = np.random.default_rng(seed)
    p = X.shape[1]
    chunk = p // (K - 1)
    weights = []
    for k in range(1, K):
        start = (k - 1) * chunk
        end = k * chunk if k < K - 1 else p
        w = np.var(X[:, start:end], axis=0).astype(np.float64)
        w += 1e-12                      # avoid zero weights
        w /= w.sum()
        weights.append((start, end, w.astype(np.float32)))
    return weights


def emax_curve(s: np.ndarray, ec50: float, emax: float = 1.0) -> np.ndarray:
    """
    Emax pharmacokinetic dose-response model:
        phi(s) = Emax * s / (EC50 + s)
    Standard in pharmacokinetics; used by DRNet for TCGA.
    """
    return emax * s / (ec50 + s + 1e-8)


def build_oracle_curves(
    X: np.ndarray,
    weights: list,
    dose_grid: np.ndarray,
    K: int,
) -> np.ndarray:
    """
    Build the noise-free potential-outcome tensor mu [N, K, M].

    Control arm (t=0):
        mu_i(0, s) = baseline_i   (constant, dose has no effect)

    Active arms (t=1,2,3):
        mu_i(t, s) = baseline_i + linear_response_i(t) * phi(s, EC50_t)

    EC50 values differ per treatment (0.2, 0.35, 0.5) so the three drugs
    have different dose-potency relationships — matching DRNet's design.
    """
    N, M = X.shape[0], len(dose_grid)
    mu = np.zeros((N, K, M), dtype=np.float32)

    # Patient-level baseline = mean normalised expression (proxy for health)
    baseline = X.mean(axis=1)            # [N]

    ec50_per_treatment = [0.20, 0.35, 0.50]   # t=1, t=2, t=3

    for m, s in enumerate(dose_grid):
        # Control: no treatment effect at any dose
        mu[:, 0, m] = baseline

        for k in range(1, K):
            start, end, w = weights[k - 1]
            linear = X[:, start:end] @ w        # [N]  dot product
            ec50 = ec50_per_treatment[k - 1]
            phi = emax_curve(np.array([s]), ec50)[0]
            mu[:, k, m] = baseline + linear * phi

    return mu


def assign_treatments(
    mu: np.ndarray,
    dose_grid: np.ndarray,
    rng: np.random.Generator,
) -> tuple:
    """
    Assignment mechanism (mirrors DRNet TCGA):

    - Propensity: softmax of potential outcomes at dose s=0.75 across K arms.
    - Each patient is assigned to the arm with highest (noisy) potential.
    - Dose: Beta(2, 2) for active treatments; 0 for control.

    Returns (t [N], d [N]).
    """
    N, K, M = mu.shape
    idx_75 = int(0.75 * M)                 # evaluation point for assignment
    potentials = mu[:, :, idx_75]           # [N, K]

    # Softmax propensity
    pot_shifted = potentials - potentials.max(axis=1, keepdims=True)
    exp_pot = np.exp(pot_shifted)
    propensity = exp_pot / exp_pot.sum(axis=1, keepdims=True)   # [N, K]

    t = np.array(
        [rng.choice(K, p=propensity[i]) for i in range(N)],
        dtype=np.int64,
    )

    d = np.zeros(N, dtype=np.float32)
    active_mask = t > 0
    n_active = int(active_mask.sum())
    if n_active > 0:
        d[active_mask] = rng.beta(2.0, 2.0, size=n_active).astype(np.float32)

    LOG.info(
        f"Treatment distribution: "
        + ", ".join(f"t={k}: {(t == k).sum()}" for k in range(K))
    )
    return t, d


def sample_factual_outcomes(
    mu: np.ndarray,
    t: np.ndarray,
    d: np.ndarray,
    dose_grid: np.ndarray,
    noise_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Factual outcome = oracle value at (t_i, d_i) + Gaussian noise.
    Oracle value is linearly interpolated on the dose grid.
    """
    N = len(t)
    y = np.zeros(N, dtype=np.float32)
    for i in range(N):
        ti, di = int(t[i]), float(d[i])
        # Linear interpolation on the dose grid
        m = np.searchsorted(dose_grid, di)
        if m == 0:
            y_noiseless = float(mu[i, ti, 0])
        elif m >= len(dose_grid):
            y_noiseless = float(mu[i, ti, -1])
        else:
            lo, hi = dose_grid[m - 1], dose_grid[m]
            w_hi = (di - lo) / (hi - lo + 1e-12)
            y_noiseless = float(mu[i, ti, m - 1]) * (1 - w_hi) + float(mu[i, ti, m]) * w_hi
        y[i] = y_noiseless + rng.normal(0.0, noise_std)
    return y


# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Convert tcga.db → curves npz for MISE evaluation")
    parser.add_argument("--db",             required=True,  help="Path to tcga.db")
    parser.add_argument("--min_val",        required=True,  help="Path to min_val.npy")
    parser.add_argument("--max_val",        required=True,  help="Path to max_val.npy")
    parser.add_argument("--out",            required=True,  help="Output .npz path")
    parser.add_argument("--num_features",   type=int, default=500,  help="Number of genes to keep (default: 500)")
    parser.add_argument("--num_grid_points",type=int, default=10,   help="Dose grid size M (default: 10)")
    parser.add_argument("--noise_std",      type=float, default=0.1, help="Outcome noise σ (default: 0.1)")
    parser.add_argument("--seed",           type=int, default=42,    help="Random seed (default: 42)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    K = 4   # 1 control + 3 active treatments (fixed, as in DRNet TCGA)

    # --- Load and prepare features ---
    min_val = np.load(args.min_val)[:-1]    # DRNet slices [:-1] (last col is outcome)
    max_val = np.load(args.max_val)[:-1]

    X_full = load_rnaseq_from_db(args.db, min_val, max_val)
    X = select_top_features(X_full, args.num_features)
    N = X.shape[0]
    LOG.info(f"Feature matrix: {X.shape}")

    # --- Build semi-synthetic benchmark ---
    dose_grid = np.linspace(0.0, 1.0, args.num_grid_points, dtype=np.float32)
    LOG.info(f"Dose grid ({args.num_grid_points} points): {dose_grid}")

    weights = build_response_weights(X, K, seed=args.seed)
    mu_all  = build_oracle_curves(X, weights, dose_grid, K)    # [N, K, M]
    LOG.info(f"Oracle curves: {mu_all.shape}   min={mu_all.min():.3f}  max={mu_all.max():.3f}")

    t, d = assign_treatments(mu_all, dose_grid, rng)
    y    = sample_factual_outcomes(mu_all, t, d, dose_grid, args.noise_std, rng)
    LOG.info(f"Factual outcomes: mean={y.mean():.3f}  std={y.std():.3f}")

    # --- Save ---
    np.savez_compressed(
        args.out,
        feature=X,         # [N, p]   covariates
        t=t,               # [N]      treatment type (0-3)
        d=d,               # [N]      observed dose
        y=y,               # [N]      factual outcome (noisy)
        eval_y=mu_all,     # [N, K, M] oracle dose-response curves (noise-free)
        eval_d=dose_grid,  # [M]      dose evaluation grid
    )
    LOG.info(f"Saved → {args.out}")
    LOG.info(
        f"\nSummary\n"
        f"  Patients  (N) : {N}\n"
        f"  Features  (p) : {X.shape[1]}\n"
        f"  Treatments(K) : {K}\n"
        f"  Grid pts  (M) : {args.num_grid_points}\n"
        f"  eval_y shape  : {mu_all.shape}   ← [N, K, M] ✓\n"
    )


if __name__ == "__main__":
    main()