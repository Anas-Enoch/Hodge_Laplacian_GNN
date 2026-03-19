import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp

# =========================
# CONFIG
# =========================

SAMPLE_ID = "GSM_6433618"
FLUX_TAG = "flux_tumor_immune"

DATA_DIR = "stats/gnn_data"
OUT_DIR = "stats"

# 🔴 DEFINE DEVICE FIRST
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD DATA
# =========================

prefix = f"{SAMPLE_ID}_{FLUX_TAG}"

print("Loading prepared GNN data (corrected paths)...")

edge_index = np.load(os.path.join(DATA_DIR, f"{prefix}_edge_index.npy"))
edge_attr = np.load(os.path.join(DATA_DIR, f"{prefix}_edge_attr.npy"))
target_flux = np.load(os.path.join(DATA_DIR, f"{prefix}_y_edge.npy"))

edge_index = torch.tensor(edge_index, dtype=torch.long).to(DEVICE)
edge_attr = torch.tensor(edge_attr, dtype=torch.float32).to(DEVICE)
target_flux = torch.tensor(target_flux, dtype=torch.float32).to(DEVICE)

num_edges = target_flux.shape[0]

# =========================
# SIMPLE GNN MODEL
# =========================

class EdgeMLP(torch.nn.Module):
    def __init__(self, in_dim=1, hidden=64):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


model = EdgeMLP(in_dim=edge_attr.shape[1]).to(DEVICE)

# =========================
# TRAINING (NO CONSTRAINT)
# =========================

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 200

print("Training WITHOUT conservation constraint...")

loss_history = []

for epoch in range(EPOCHS):

    model.train()
    optimizer.zero_grad()

    pred_flux = model(edge_attr)

    # ONLY data loss
    loss = torch.mean((pred_flux - target_flux) ** 2)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.6f}")

# =========================
# SAVE MODEL OUTPUT
# =========================

pred_flux_np = pred_flux.detach().cpu().numpy()

out_flux_path = os.path.join(
    OUT_DIR,
    f"{SAMPLE_ID}_step18_unconstrained_flux_{FLUX_TAG}.csv"
)

pd.DataFrame({"flux": pred_flux_np}).to_csv(out_flux_path, index=False)

print(f"Saved unconstrained flux → {out_flux_path}")

# =========================
# LOAD HODGE COMPONENTS
# =========================

print("Loading Hodge decomposition matrices...")

B1 = sp.load_npz(f"stats/gnn_data/{prefix}_B1.npz").toarray()
B2 = sp.load_npz(f"stats/gnn_data/{prefix}_B2.npz").toarray()

# Convert to torch
B1 = torch.tensor(B1, dtype=torch.float32).to(DEVICE)
B2 = torch.tensor(B2, dtype=torch.float32).to(DEVICE)

# =========================
# HODGE DECOMPOSITION (SAFE)
# =========================

f = torch.tensor(pred_flux_np, dtype=torch.float32).to(DEVICE)

# --- Exact ---
if B1.shape[0] != f.shape[0]:
    B1 = B1.T

phi = torch.linalg.lstsq(B1, f).solution
f_exact = B1 @ phi

# --- Coexact ---
B2T = B2.T

if B2T.shape[0] != f.shape[0]:
    B2T = B2T.T

psi = torch.linalg.lstsq(B2T, f).solution
f_coexact = B2T @ psi

# --- Harmonic ---
f_harmonic = f - f_exact - f_coexact

# Energies
E_exact = torch.norm(f_exact).item()
E_coexact = torch.norm(f_coexact).item()
E_harmonic = torch.norm(f_harmonic).item()
E_total = torch.norm(f).item()

# Normalize
E_exact /= E_total
E_coexact /= E_total
E_harmonic /= E_total

# =========================
# SAVE ENERGY SUMMARY
# =========================

energy_df = pd.DataFrame({
    "component": ["exact", "coexact", "harmonic"],
    "energy_fraction": [E_exact, E_coexact, E_harmonic]
})

energy_path = os.path.join(
    OUT_DIR,
    f"{SAMPLE_ID}_step18_unconstrained_energy_{FLUX_TAG}.csv"
)

energy_df.to_csv(energy_path, index=False)

print(f"Saved energy summary → {energy_path}")

# =========================
# LOAD CONSTRAINED RESULT
# =========================

constrained_path = os.path.join(
    OUT_DIR,
    f"{SAMPLE_ID}_step15_gnn_operator_summary_{FLUX_TAG}.csv"
)

constrained_df = pd.read_csv(constrained_path)

# =========================
# ROBUST COLUMN HANDLING
# =========================

print("Constrained DF columns:", constrained_df.columns)

if all(c in constrained_df.columns for c in ["frac_exact", "frac_coexact", "frac_harmonic"]):
    constrained_vals = constrained_df.loc[0, ["frac_exact", "frac_coexact", "frac_harmonic"]].to_numpy(dtype=float)

elif "energy_fraction" in constrained_df.columns:
    constrained_vals = constrained_df["energy_fraction"].to_numpy(dtype=float)

elif "energy" in constrained_df.columns:
    constrained_vals = constrained_df["energy"].to_numpy(dtype=float)

else:
    raise ValueError(
        f"Could not find constrained energy-fraction columns. Found columns: {list(constrained_df.columns)}"
    )

comparison_df = pd.DataFrame({
    "component": ["exact", "coexact", "harmonic"],
    "constrained": constrained_vals,
    "unconstrained": energy_df["energy_fraction"].to_numpy(dtype=float)
})

comparison_path = os.path.join(
    OUT_DIR,
    f"{SAMPLE_ID}_step18_ablation_comparison_{FLUX_TAG}.csv"
)

comparison_df.to_csv(comparison_path, index=False)

print(f"Saved comparison → {comparison_path}")

# =========================
# PLOT
# =========================

plt.figure(figsize=(6, 4))

x = np.arange(3)

plt.bar(x - 0.15, comparison_df["constrained"], width=0.3, label="Constrained")
plt.bar(x + 0.15, comparison_df["unconstrained"], width=0.3, label="Unconstrained")

plt.xticks(x, ["Exact", "Coexact", "Harmonic"])
plt.ylabel("Energy fraction")
plt.title("Ablation: Effect of conservation constraint")
plt.legend()

plot_path = os.path.join(
    OUT_DIR,
    f"{SAMPLE_ID}_step18_ablation_plot_{FLUX_TAG}.png"
)

plt.tight_layout()
plt.savefig(plot_path, dpi=300)

print(f"Saved plot → {plot_path}")

print("STEP 18 COMPLETE.")
