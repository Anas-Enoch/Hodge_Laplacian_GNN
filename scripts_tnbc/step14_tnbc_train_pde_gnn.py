from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import sparse


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path


def load_npz_to_torch_dense(path: Path, device: torch.device) -> torch.Tensor:
    mat = sparse.load_npz(path).astype(np.float32)
    return torch.tensor(mat.toarray(), dtype=torch.float32, device=device)


def load_numpy(path: Path, device: torch.device, dtype=torch.float32) -> torch.Tensor:
    arr = np.load(path, allow_pickle=True)
    return torch.tensor(arr, dtype=dtype, device=device)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan
    c = np.corrcoef(a, b)[0, 1]
    return float(c) if np.isfinite(c) else np.nan


class NodeEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.lin1 = nn.Linear(in_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        h = A_norm @ x
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)

        h = A_norm @ h
        h = self.lin2(h)
        h = F.relu(h)
        return h


class AntiSymEdgeDecoder(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def phi(self, hi: torch.Tensor, hj: torch.Tensor, eattr: torch.Tensor) -> torch.Tensor:
        z = torch.cat([hi, hj, eattr], dim=-1)
        return self.mlp(z).squeeze(-1)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        tail = edge_index[0]
        head = edge_index[1]

        hi = h[tail]
        hj = h[head]

        forward_val = self.phi(hi, hj, edge_attr)
        backward_val = self.phi(hj, hi, edge_attr)

        # antisymmetric by construction
        flux = forward_val - backward_val
        return flux


class PDEConstrainedEdgeGNN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        hidden_node: int = 32,
        latent_node: int = 32,
        hidden_edge: int = 32,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = NodeEncoder(
            in_dim=in_dim,
            hidden_dim=hidden_node,
            out_dim=latent_node,
            dropout=dropout,
        )
        self.decoder = AntiSymEdgeDecoder(
            node_dim=latent_node,
            edge_dim=edge_dim,
            hidden_dim=hidden_edge,
        )

    def forward(
        self,
        x: torch.Tensor,
        A_norm: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        h = self.encoder(x, A_norm)
        f = self.decoder(h, edge_index, edge_attr)
        return f


def build_node_adjacency_from_B1(B1: torch.Tensor) -> torch.Tensor:
    B1_abs = torch.abs(B1)
    A = B1_abs @ B1_abs.T
    A = A - torch.diag(torch.diag(A))
    A = (A > 0).float()
    return A


def normalize_adjacency(A: torch.Tensor) -> torch.Tensor:
    I = torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
    A_hat = A + I
    deg = A_hat.sum(dim=1)
    deg_inv_sqrt = torch.pow(torch.clamp(deg, min=1.0), -0.5)
    D_inv_sqrt = torch.diag(deg_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    y_edge_scaled: torch.Tensor,
    B1: torch.Tensor,
    L1: torch.Tensor,
    A_norm: torch.Tensor,
    *,
    epochs: int,
    lr: float,
    weight_decay: float,
    lambda_data: float,
    lambda_div: float,
    lambda_smooth: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        f_pred_scaled = model(x, A_norm, edge_index, edge_attr)

        loss_data = torch.mean((f_pred_scaled - y_edge_scaled) ** 2)
        div = B1 @ f_pred_scaled
        loss_div = torch.mean(div ** 2)
        loss_smooth = torch.dot(f_pred_scaled, L1 @ f_pred_scaled) / f_pred_scaled.numel()

        loss = (
            lambda_data * loss_data
            + lambda_div * loss_div
            + lambda_smooth * loss_smooth
        )

        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred_np = f_pred_scaled.detach().cpu().numpy()
            targ_np = y_edge_scaled.detach().cpu().numpy()

            mse = float(np.mean((pred_np - targ_np) ** 2))
            mae = float(np.mean(np.abs(pred_np - targ_np)))
            corr = safe_corr(pred_np, targ_np)
            pred_std = float(np.std(pred_np))
            targ_std = float(np.std(targ_np))

        history.append(
            {
                "epoch": epoch,
                "loss_total": float(loss.item()),
                "loss_data": float(loss_data.item()),
                "loss_div": float(loss_div.item()),
                "loss_smooth": float(loss_smooth.item()),
                "mse_to_target_scaled": mse,
                "mae_to_target_scaled": mae,
                "corr_to_target_scaled": corr,
                "std_pred_scaled": pred_std,
                "std_target_scaled": targ_std,
            }
        )

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(
                f"[epoch {epoch:4d}] "
                f"total={loss.item():.6e}  "
                f"data={loss_data.item():.6e}  "
                f"div={loss_div.item():.6e}  "
                f"smooth={loss_smooth.item():.6e}  "
                f"corr_scaled={corr:.4f}  "
                f"std_pred={pred_std:.4f}"
            )

    model.eval()
    with torch.no_grad():
        f_final_scaled = model(x, A_norm, edge_index, edge_attr).detach().cpu().numpy()

    return f_final_scaled, pd.DataFrame(history)


def save_history_plot(history_df: pd.DataFrame, outpath: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].plot(history_df["epoch"], history_df["loss_total"], label="total")
    axes[0].plot(history_df["epoch"], history_df["loss_data"], label="data")
    axes[0].plot(history_df["epoch"], history_df["loss_div"], label="div")
    axes[0].plot(history_df["epoch"], history_df["loss_smooth"], label="smooth")
    axes[0].set_title("Training losses")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["corr_to_target_scaled"], label="corr_scaled")
    axes[1].plot(history_df["epoch"], history_df["mae_to_target_scaled"], label="mae_scaled")
    axes[1].plot(history_df["epoch"], history_df["std_pred_scaled"], label="std_pred_scaled")
    axes[1].set_title("Fit to standardized target")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 14 TNBC: train PDE-constrained edge GNN against standardized residualized proxy flux."
    )
    parser.add_argument("--sample_id", required=True)
    parser.add_argument(
        "--target_flux",
        required=True,
        choices=["flux_tumor_immune", "flux_tumor_stroma", "flux_immune_stroma"],
    )
    parser.add_argument("--datadir", default="stats/gnn_data")
    parser.add_argument("--statsdir", default="stats")
    parser.add_argument("--figdir", default="visium_figures")

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    parser.add_argument("--hidden_node", type=int, default=32)
    parser.add_argument("--latent_node", type=int, default=32)
    parser.add_argument("--hidden_edge", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--lambda_data", type=float, default=10.0)
    parser.add_argument("--lambda_div", type=float, default=0.01)
    parser.add_argument("--lambda_smooth", type=float, default=0.001)

    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sample_id = args.sample_id
    target_flux = args.target_flux
    prefix = f"{sample_id}_{target_flux}"

    datadir = Path(args.datadir)
    statsdir = Path(args.statsdir)
    figdir = Path(args.figdir)
    statsdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    X_path = require_file(datadir / f"{prefix}_X.npy")
    edge_index_path = require_file(datadir / f"{prefix}_edge_index.npy")
    edge_attr_path = require_file(datadir / f"{prefix}_edge_attr.npy")
    y_edge_path = require_file(datadir / f"{prefix}_y_edge.npy")
    B1_path = require_file(datadir / f"{prefix}_B1.npz")
    L1_path = require_file(datadir / f"{prefix}_L1.npz")
    edge_meta_csv = require_file(datadir / f"{prefix}_edges_for_gnn.csv")

    X = load_numpy(X_path, device=device, dtype=torch.float32)
    edge_index = load_numpy(edge_index_path, device=device, dtype=torch.long)
    edge_attr = load_numpy(edge_attr_path, device=device, dtype=torch.float32)

    y_edge_unscaled = np.load(y_edge_path, allow_pickle=True).astype(np.float32)
    y_mean = float(np.mean(y_edge_unscaled))
    y_std = float(np.std(y_edge_unscaled))
    if y_std < 1e-12:
        raise RuntimeError("Target edge flux has near-zero variance; cannot standardize safely.")

    y_edge_scaled_np = ((y_edge_unscaled - y_mean) / y_std).astype(np.float32)
    y_edge_scaled = torch.tensor(y_edge_scaled_np, dtype=torch.float32, device=device)

    B1 = load_npz_to_torch_dense(B1_path, device=device)
    L1 = load_npz_to_torch_dense(L1_path, device=device)

    A = build_node_adjacency_from_B1(B1)
    A_norm = normalize_adjacency(A)

    edge_meta = pd.read_csv(edge_meta_csv)

    model = PDEConstrainedEdgeGNN(
        in_dim=X.shape[1],
        edge_dim=edge_attr.shape[1],
        hidden_node=args.hidden_node,
        latent_node=args.latent_node,
        hidden_edge=args.hidden_edge,
        dropout=args.dropout,
    ).to(device)

    print("=" * 80)
    print("STEP 14: PDE-constrained GNN training (standardized target)")
    print("=" * 80)
    print(f"sample_id        : {sample_id}")
    print(f"target_flux      : {target_flux}")
    print(f"device           : {device}")
    print(f"n_nodes          : {X.shape[0]}")
    print(f"n_edges          : {edge_index.shape[1]}")
    print(f"n_features       : {X.shape[1]}")
    print(f"edge_attr_dim    : {edge_attr.shape[1]}")
    print(f"epochs           : {args.epochs}")
    print(f"lr               : {args.lr}")
    print(f"lambda_data      : {args.lambda_data}")
    print(f"lambda_div       : {args.lambda_div}")
    print(f"lambda_smooth    : {args.lambda_smooth}")
    print(f"target mean      : {y_mean:.6e}")
    print(f"target std       : {y_std:.6e}")
    print("-" * 80)

    f_pred_scaled, history_df = train_model(
        model=model,
        x=X,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y_edge_scaled=y_edge_scaled,
        B1=B1,
        L1=L1,
        A_norm=A_norm,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_data=args.lambda_data,
        lambda_div=args.lambda_div,
        lambda_smooth=args.lambda_smooth,
    )

    # Unscale back to physical edge flux units
    f_pred_unscaled = f_pred_scaled * y_std + y_mean

    learned_df = edge_meta.copy()
    learned_df["target_flux_unscaled"] = y_edge_unscaled
    learned_df["target_flux_scaled"] = y_edge_scaled_np
    learned_df["flux_gnn_scaled"] = f_pred_scaled
    learned_df["flux_gnn_unscaled"] = f_pred_unscaled
    learned_df["residual_scaled"] = learned_df["flux_gnn_scaled"] - learned_df["target_flux_scaled"]
    learned_df["residual_unscaled"] = learned_df["flux_gnn_unscaled"] - learned_df["target_flux_unscaled"]

    learned_csv = statsdir / f"{sample_id}_step14_gnn_learned_flux_{target_flux}.csv"
    learned_df.to_csv(learned_csv, index=False)

    history_csv = statsdir / f"{sample_id}_step14_gnn_training_history_{target_flux}.csv"
    history_df.to_csv(history_csv, index=False)

    history_png = figdir / f"{sample_id}_step14_gnn_training_history_{target_flux}.png"
    save_history_plot(history_df, history_png)

    ckpt = {
        "sample_id": sample_id,
        "target_flux": target_flux,
        "model_state_dict": model.state_dict(),
        "target_mean": y_mean,
        "target_std": y_std,
        "config": vars(args),
    }
    ckpt_path = statsdir / f"{sample_id}_step14_gnn_model_{target_flux}.pt"
    torch.save(ckpt, ckpt_path)

    corr_scaled = safe_corr(
        learned_df["flux_gnn_scaled"].to_numpy(),
        learned_df["target_flux_scaled"].to_numpy(),
    )
    corr_unscaled = safe_corr(
        learned_df["flux_gnn_unscaled"].to_numpy(),
        learned_df["target_flux_unscaled"].to_numpy(),
    )

    div_pred_unscaled = (
        B1 @ torch.tensor(f_pred_unscaled, dtype=torch.float32, device=device)
    ).detach().cpu().numpy()
    div_target_unscaled = (
        B1 @ torch.tensor(y_edge_unscaled, dtype=torch.float32, device=device)
    ).detach().cpu().numpy()

    summary = pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "target_flux": target_flux,
                "n_nodes": int(X.shape[0]),
                "n_edges": int(edge_index.shape[1]),
                "target_mean": y_mean,
                "target_std": y_std,
                "final_mse_scaled": float(np.mean((learned_df["flux_gnn_scaled"] - learned_df["target_flux_scaled"]) ** 2)),
                "final_mae_scaled": float(np.mean(np.abs(learned_df["flux_gnn_scaled"] - learned_df["target_flux_scaled"]))),
                "final_corr_scaled": corr_scaled,
                "final_mse_unscaled": float(np.mean((learned_df["flux_gnn_unscaled"] - learned_df["target_flux_unscaled"]) ** 2)),
                "final_mae_unscaled": float(np.mean(np.abs(learned_df["flux_gnn_unscaled"] - learned_df["target_flux_unscaled"]))),
                "final_corr_unscaled": corr_unscaled,
                "mean_abs_div_gnn_unscaled": float(np.mean(np.abs(div_pred_unscaled))),
                "mean_abs_div_target_unscaled": float(np.mean(np.abs(div_target_unscaled))),
                "std_flux_gnn_scaled": float(np.std(learned_df["flux_gnn_scaled"])),
                "std_flux_target_scaled": float(np.std(learned_df["target_flux_scaled"])),
                "std_flux_gnn_unscaled": float(np.std(learned_df["flux_gnn_unscaled"])),
                "std_flux_target_unscaled": float(np.std(learned_df["target_flux_unscaled"])),
            }
        ]
    )
    summary_csv = statsdir / f"{sample_id}_step14_gnn_summary_{target_flux}.csv"
    summary.to_csv(summary_csv, index=False)

    print("\nFinal summary")
    print("-" * 80)
    print(summary.to_string(index=False))
    print("-" * 80)
    print(f"Saved: {learned_csv}")
    print(f"Saved: {history_csv}")
    print(f"Saved: {history_png}")
    print(f"Saved: {ckpt_path}")
    print(f"Saved: {summary_csv}")


if __name__ == "__main__":
    main()
