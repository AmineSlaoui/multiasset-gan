"""Generate synthetic returns for the test period and save to CSV.

Run from PCAFactor_Model/:
    python generate_test_returns.py

Output: data/generated_returns_test.csv  (T_test x 60 assets)
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from sfmg_generator import SFMGGenerator

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE      = os.path.dirname(__file__)
DATA_DIR  = os.path.join(BASE, "data")
CKPT_PATH = os.path.join(BASE, "results", "sfmg_sector_seed42", "best_model.pt")
CFG_PATH  = os.path.join(BASE, "results", "sfmg_sector_seed42", "config.json")
OUT_PATH  = os.path.join(DATA_DIR, "generated_returns_test.csv")

# ── Load pre-computed arrays ──────────────────────────────────────────────────
meta       = json.load(open(os.path.join(DATA_DIR, "meta.json")))
cov_arr    = np.load(os.path.join(DATA_DIR, "cov_arr.npy"))          # (T, d_cov)
f_arr      = np.load(os.path.join(DATA_DIR, "f_arr_lag1.npy"))       # (T, K)
alpha_hat  = np.load(os.path.join(DATA_DIR, "alpha_hat.npy"))        # (T, N)
beta_hat   = np.load(os.path.join(DATA_DIR, "beta_hat.npy"))         # (T, N, K)
sigma_hat  = np.load(os.path.join(DATA_DIR, "sigma_hat.npy"))        # (T, N)
L_eps      = np.load(os.path.join(DATA_DIR, "residual_cholesky.npy"))# (N, N)

returns_df = pd.read_csv(
    os.path.join(DATA_DIR, "all_assets_log_returns.csv"),
    index_col="Date", parse_dates=True,
)

asset_list      = meta["asset_list"]              # 60 asset names
test_start_idx  = meta["dates"]["test_start_idx"] # 3522
test_end_idx    = meta["dates"]["test_end_idx"]   # 3773
T_test          = test_end_idx - test_start_idx
N_ASSETS        = meta["n_assets"]                # 60
N_FACTORS       = meta["n_factors"]               # 11
D_COV           = meta["d_cov"]                   # 15

print(f"Test period: idx {test_start_idx} → {test_end_idx}  ({T_test} days)")
print(f"Assets: {N_ASSETS},  Factors: {N_FACTORS},  d_cov: {D_COV}")

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

cfg = json.load(open(CFG_PATH))
gen = SFMGGenerator(
    n_assets=N_ASSETS, n_factors=N_FACTORS, d_z=10, d_cov=D_COV,
    hidden_dim=cfg["hidden_dim"], num_blocks=cfg["num_blocks"], dropout=0.2,
    eta_low=cfg["eta_low"],       eta_high=cfg["eta_high"],
    eta_sigma_low=cfg["eta_sigma_low"], eta_sigma_high=cfg["eta_sigma_high"],
    residual_rank=cfg["residual_rank"],
    residual_cholesky=L_eps,
    trainable_L_rho=cfg.get("trainable_L_rho", False),
).to(device)

ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
gen.load_state_dict(ckpt["G"], strict=False)
gen.eval()
print("Checkpoint loaded.")

# ── Generation ────────────────────────────────────────────────────────────────
N_PATHS = 200
BATCH   = 20
rfs     = gen.rfs                       # receptive field size
T_FULL  = rfs + T_test + 1

t_start = max(0, test_start_idx - rfs - 1)
t_end   = min(t_start + T_FULL, len(cov_arr))
cov_seq = torch.tensor(cov_arr[t_start:t_end], dtype=torch.float32).T.unsqueeze(0).to(device)
if t_end - t_start < T_FULL:
    cov_seq = F.pad(cov_seq, (0, T_FULL - (t_end - t_start)), mode="replicate")

ts, te = test_start_idx, test_end_idx
f  = torch.tensor(f_arr[ts:te],      dtype=torch.float32).T.unsqueeze(0).to(device)   # (1, K, T)
a  = torch.tensor(alpha_hat[ts:te].T, dtype=torch.float32).unsqueeze(0).to(device)    # (1, N, T)
s  = torch.tensor(sigma_hat[ts:te].T, dtype=torch.float32).unsqueeze(0).to(device)    # (1, N, T)
b  = torch.tensor(beta_hat[ts:te],    dtype=torch.float32).permute(1, 2, 0).unsqueeze(0).to(device)  # (1, N, K, T)

all_paths = np.zeros((N_PATHS, T_test, N_ASSETS))
print(f"Generating {N_PATHS} paths...")
for i in range(0, N_PATHS, BATCH):
    n = min(BATCH, N_PATHS - i)
    z = torch.randn(n, gen.d_z, T_FULL, device=device)
    with torch.no_grad():
        r = gen(
            z,
            cov_seq.expand(n, -1, -1),
            a.expand(n, -1, -1),
            b.expand(n, -1, -1, -1),
            s.expand(n, -1, -1),
            f.expand(n, -1, -1),
        )
    for j in range(n):
        all_paths[i + j] = r[j, :, :T_test].cpu().numpy().T
    print(f"  {i + n}/{N_PATHS} paths done")

# ── Save mean path as CSV ─────────────────────────────────────────────────────
mean_returns = all_paths.mean(axis=0)   # (T_test, N)

test_dates = returns_df.index[test_start_idx:test_end_idx]
out_df = pd.DataFrame(mean_returns, index=test_dates, columns=asset_list)
out_df.index.name = "Date"
out_df.to_csv(OUT_PATH)
print(f"\nSaved: {OUT_PATH}  shape={out_df.shape}")

# Also save all paths as npy for deeper analysis
npy_path = os.path.join(DATA_DIR, "generated_paths_test.npy")
np.save(npy_path, all_paths)
print(f"Saved all paths: {npy_path}  shape={all_paths.shape}")