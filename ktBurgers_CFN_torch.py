import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class FluxNet(nn.Module):
    def __init__(self, features: List[int]):
        super().__init__()
        if len(features) < 2:
            raise ValueError("features must include input and output sizes, e.g. [64, ..., 1]")
        layers: List[nn.Module] = []
        for in_f, out_f in zip(features[:-1], features[1:]):
            layers.append(nn.Linear(in_f, out_f))
            if out_f != features[-1]:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)

    def forward(self, conservative_variables: torch.Tensor) -> torch.Tensor:
        # conservative_variables: (..., C)
        return self.net(conservative_variables)


class KurganovTadmorSchemeTorch(nn.Module):
    def __init__(
        self,
        features: List[int],
        dt: float,
        dx: float,
        boundary: str = "same",
        limiter: str = "minmod",
    ):
        super().__init__()
        self.num_flux = FluxNet(features)
        self.dt = float(dt)
        self.dx = float(dx)
        self.boundary = boundary.lower()
        self.limiter = limiter.lower()

    def _minmod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Matches the JAX implementation in ktBurgers_CFN.py
        zeros = torch.zeros_like(a)
        return torch.sign(b) * torch.maximum(
            zeros,
            torch.minimum(torch.abs(b), torch.sign(b) * a),
        )

    def linear_extrapolation(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        u: (B, N+4, C)
        returns:
          uL, uR: (B, N+1, C)
        """
        um = u[:, :-2, :]
        u_ = u[:, 1:-1, :]
        up = u[:, 2:, :]

        slope = self._minmod(u_ - um, up - u_)
        uL = u_ + 0.5 * slope
        uR = u_ - 0.5 * slope
        return uL[:, :-1, :], uR[:, 1:, :]

    def flux_and_dflux(self, up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        up: (B, Np, C)
        returns:
          f(up): (B, Np, C)
          df/dup: (B, Np, C)

        Note: this uses autograd to compute df/du. During training we keep the higher-order
        graph so gradients can flow to flux parameters through df/du. During eval/validation
        we avoid higher-order graphs.
        """
        # Validation/eval often wraps forward passes in torch.no_grad(), but we still need
        # first-order gradients w.r.t. inputs to estimate wave speeds.
        with torch.enable_grad():
            up_leaf = up.detach().requires_grad_(True)
            b, n, c = up_leaf.shape
            f = self.num_flux(up_leaf.reshape(b * n, c)).reshape(b, n, -1)
            df = torch.autograd.grad(f.sum(), up_leaf, create_graph=self.training)[0]
        return f, df

    def kurganov_tadmor(self, u_padded: torch.Tensor) -> torch.Tensor:
        """
        u_padded: (B, N+4, C)
        returns rhs: (B, N, C)
        """
        uL, uR = self.linear_extrapolation(u_padded)
        fL, dfL = self.flux_and_dflux(uL)
        fR, dfR = self.flux_and_dflux(uR)

        rho = torch.maximum(torch.abs(dfL), torch.abs(dfR))
        H = 0.5 * (fR + fL - rho * (uR - uL))
        return -(H[:, 1:, :] - H[:, :-1, :]) / self.dx

    def rhs(self, u: torch.Tensor) -> torch.Tensor:
        # Periodic wrap padding by 2 cells on each side (matches jnp.pad(..., mode="wrap")).
        if self.boundary != "same":
            raise NotImplementedError(f"Only boundary='same' (periodic wrap) supported, got {self.boundary!r}")
        u_padded = torch.cat([u[:, -2:, :], u, u[:, :2, :]], dim=1)
        return self.kurganov_tadmor(u_padded)

    def tvd_rk3(self, u: torch.Tensor) -> torch.Tensor:
        u1 = u + self.dt * self.rhs(u)
        u2 = 0.75 * u + 0.25 * u1 + 0.25 * self.dt * self.rhs(u1)
        u3 = (1.0 / 3.0) * u + (2.0 / 3.0) * u2 + (2.0 / 3.0) * self.dt * self.rhs(u2)
        return u3

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.tvd_rk3(u)


class BurgersRolloutDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        rollout_steps: int,
        noise_level: float,
        seed: int = 0,
        index: int = 0,
        dtype: np.dtype = np.float32,
    ):
        super().__init__()
        arr = np.load(data_path)
        if arr.ndim != 4:
            raise ValueError(f"Expected data array with ndim=4 (S,T,N,C), got {arr.shape}")
        if index + rollout_steps >= arr.shape[1]:
            raise ValueError(
                f"rollout_steps={rollout_steps} too large for T={arr.shape[1]} at index={index}"
            )

        un = arr[:, index, :, :].astype(dtype, copy=False)
        un_p1 = arr[:, index + 1 : index + 1 + rollout_steps, :, :].astype(dtype, copy=False)

        if noise_level and noise_level > 0:
            rng = np.random.default_rng(seed)
            scale = float(np.mean(np.abs(arr))) * float(noise_level)
            un_p1 = un_p1 + rng.normal(size=un_p1.shape).astype(dtype) * scale

        self.un = un
        self.un_p1 = un_p1

    def __len__(self) -> int:
        return self.un.shape[0]

    def __getitem__(self, idx: int):
        # Return torch tensors with shape:
        #   un: (N, C)
        #   un_p1: (L, N, C)
        return torch.from_numpy(self.un[idx]), torch.from_numpy(self.un_p1[idx])


@dataclass
class TrainConfig:
    nx: int = 512
    dt: float = 0.005
    steps: int = 20
    batch_size: int = 10
    epochs: int = 500
    lr: float = 1e-4
    noise_level: float = 1.0
    decay_steps: int = 2000
    decay_rate: float = 0.95
    end_lr: float = 1e-6


def multistep_rollout_loss(model: nn.Module, un: torch.Tensor, u_np1: torch.Tensor) -> torch.Tensor:
    """
    un: (B, N, C)
    u_np1: (B, L, N, C)
    """
    um = un
    total = 0.0
    for i in range(u_np1.shape[1]):
        u_pred = model(um)
        total = total + F.mse_loss(u_pred, u_np1[:, i, :, :])
        um = u_pred
    return total


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def lr_schedule(step: int, cfg: TrainConfig) -> float:
    # Exponential decay with a floor, similar to optax.exponential_decay.
    lr = cfg.lr * (cfg.decay_rate ** (step / float(cfg.decay_steps)))
    return max(float(cfg.end_lr), float(lr))


def sanity_check_npy(data_path: str) -> None:
    arr = np.load(data_path)
    print(f"Loaded {data_path} with shape={arr.shape}, dtype={arr.dtype}")
    finite = np.isfinite(arr).all()
    print(f"finite={finite}, min={np.nanmin(arr):.6g}, max={np.nanmax(arr):.6g}, mean(abs)={np.mean(np.abs(arr)):.6g}")


def train(
    train_path: str,
    val_path: str,
    ckpt_path: str,
    cfg: TrainConfig,
    device: torch.device,
    features: Optional[List[int]] = None,
) -> nn.Module:
    if features is None:
        # Input/Output channel count is 1 for Burgers here.
        features = [1, 64, 64, 64, 64, 64, 1]

    dx = 2 * np.pi / cfg.nx

    train_ds = BurgersRolloutDataset(train_path, rollout_steps=cfg.steps, noise_level=cfg.noise_level, seed=0)
    val_ds = BurgersRolloutDataset(val_path, rollout_steps=cfg.steps, noise_level=cfg.noise_level, seed=1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = KurganovTadmorSchemeTorch(features=features, dt=cfg.dt, dx=dx).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    best_val = float("inf")
    global_step = 0

    os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        train_losses = []

        for un, un_p1 in train_loader:
            un = un.to(device=device, dtype=torch.float32)
            un_p1 = un_p1.to(device=device, dtype=torch.float32)

            # Add batch dimension shape: (B, N, C) and (B, L, N, C)
            if un.ndim == 2:
                un = un.unsqueeze(0)
            if un_p1.ndim == 3:
                un_p1 = un_p1.unsqueeze(0)

            lr = lr_schedule(global_step, cfg)
            set_optimizer_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)
            loss = multistep_rollout_loss(model, un, un_p1)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().cpu().item())
            global_step += 1

        model.eval()
        with torch.no_grad():
            val_losses = []
            for un, un_p1 in val_loader:
                un = un.to(device=device, dtype=torch.float32)
                un_p1 = un_p1.to(device=device, dtype=torch.float32)
                if un.ndim == 2:
                    un = un.unsqueeze(0)
                if un_p1.ndim == 3:
                    un_p1 = un_p1.unsqueeze(0)
                val_losses.append(multistep_rollout_loss(model, un, un_p1).cpu().item())
            val_loss = float(np.mean(val_losses)) if val_losses else float("nan")

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        print(f"epoch {epoch:4d} | train_loss {train_loss:.10f} | val_loss {val_loss:.10f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "cfg": cfg.__dict__,
                    "features": features,
                },
                ckpt_path,
            )

    return model


def load_model(ckpt_path: str, device: torch.device, dt: float, dx: float) -> KurganovTadmorSchemeTorch:
    ckpt = torch.load(ckpt_path, map_location=device)
    features = ckpt.get("features")
    if features is None:
        raise ValueError("Checkpoint missing 'features'")
    model = KurganovTadmorSchemeTorch(features=features, dt=dt, dx=dx).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate(
    ckpt_path: str,
    test_path: str,
    out_dir: str,
    nx: int,
    dt: float,
    device: torch.device,
    n_steps: int = 600,
) -> None:
    """Rough PyTorch equivalent of evaluateESS: rollout and save a few plots."""
    import matplotlib.pyplot as plt

    dx = 2 * np.pi / nx
    model = load_model(ckpt_path, device=device, dt=dt, dx=dx)

    test_data = np.load(test_path)
    if test_data.ndim != 4:
        raise ValueError(f"Expected test data with ndim=4 (S,T,N,C), got {test_data.shape}")
    if test_data.shape[2] != nx:
        raise ValueError(f"nx mismatch: expected N={nx}, but test data has N={test_data.shape[2]}")

    # Use the first trajectory.
    un0 = torch.from_numpy(test_data[:1, 0, :, :]).to(device=device, dtype=torch.float32)

    u_rollout = [un0]
    flux_rollout = []
    with torch.no_grad():
        f0 = model.num_flux(un0.reshape(-1, un0.shape[-1])).reshape_as(un0)
        flux_rollout.append(f0)
        un = un0
        for _ in range(n_steps):
            un = model(un)
            u_rollout.append(un)
            f = model.num_flux(un.reshape(-1, un.shape[-1])).reshape_as(un)
            flux_rollout.append(f)

    x = np.linspace(-np.pi, np.pi, nx)
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    u_np = torch.stack(u_rollout, dim=1).detach().cpu().numpy()  # (1, n_steps+1, N, C)
    flux_np = torch.stack(flux_rollout, dim=1).detach().cpu().numpy()

    # Directory structure mirrors the original script.
    _ensure_dir(out_dir)
    u_dir = os.path.join(out_dir, "u")
    ent_dir = os.path.join(out_dir, "Entropy")
    cons_dir = os.path.join(out_dir, "Conserved_u")
    _ensure_dir(u_dir)
    _ensure_dir(ent_dir)
    _ensure_dir(cons_dir)

    # Snapshot plot at final time.
    j = n_steps
    pred = u_np[0, j, :, 0]
    exact = test_data[0, j, :, 0] if test_data.shape[1] > j else None
    plt.figure(figsize=(4, 3))
    if exact is not None:
        plt.plot(x, exact, label="exact")
    plt.plot(x, pred, "-.", label="pred", lw=0.8)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(f"t = {j * dt}s")
    plt.tight_layout()
    plt.savefig(os.path.join(u_dir, f"{j:03d}.png"), dpi=200)
    plt.close()

    # Entropy: mean(u^2)/2.
    if test_data.shape[1] >= n_steps + 1:
        exact_entropy = [float(np.mean(test_data[0, s, :, 0] ** 2) / 2.0) for s in range(n_steps + 1)]
    else:
        exact_entropy = None
    pred_entropy = [float(np.mean(u_np[0, s, :, 0] ** 2) / 2.0) for s in range(n_steps + 1)]

    plt.figure(figsize=(4, 3))
    if exact_entropy is not None:
        plt.plot(t, exact_entropy, label="Exact")
    plt.plot(t, pred_entropy, "-.", label="Pred")
    plt.xlabel("t / s")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(ent_dir, "entropy.png"), dpi=200)
    plt.close()

    # Conserved quantity (as in the original script).
    if test_data.shape[1] >= n_steps + 1:
        exact_cons = [float(dx * np.sum(test_data[0, s, :, 0] - test_data[0, 0, :, 0])) for s in range(n_steps + 1)]
    else:
        exact_cons = None

    pred_cons = []
    for s in range(n_steps + 1):
        flux_left = float(flux_np[0, s, 0, 0])
        flux_right = float(flux_np[0, s, -1, 0])
        pred_cons.append(float(dx * np.sum(u_np[0, s, :, 0] - u_np[0, 0, :, 0]) - dt * (flux_left - flux_right) * s))

    plt.figure(figsize=(4, 3))
    if exact_cons is not None:
        plt.plot(t, exact_cons, label="Exact")
    plt.plot(t, pred_cons, "-.", label="Pred")
    plt.xlabel("t / s")
    plt.ylabel("Conserved_U")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cons_dir, "conserved_u.png"), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "check", "eval"], default="train")
    parser.add_argument("--train", default="Data/trainData_Burgers_512.npy")
    parser.add_argument("--val", default="Data/valData_Burgers_512.npy")
    parser.add_argument("--test", default="Data/testData_Burgers_512_Low.npy")
    parser.add_argument("--ckpt", default="ckpts/kt_torch_best.pt")
    parser.add_argument("--out", default="_plots/kt_torch")
    parser.add_argument("--eval_steps", type=int, default=600)

    parser.add_argument("--nx", type=int, default=512)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--noise", type=float, default=1.0)

    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])

    args = parser.parse_args()

    if args.mode == "check":
        sanity_check_npy(args.train)
        sanity_check_npy(args.val)
        return

    if args.mode == "eval":
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested but CUDA is not available")
        if args.device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("--device=mps requested but MPS is not available")
        device = torch.device(args.device)
        evaluate(
            ckpt_path=args.ckpt,
            test_path=args.test,
            out_dir=args.out,
            nx=args.nx,
            dt=args.dt,
            device=device,
            n_steps=args.eval_steps,
        )
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device=cuda requested but CUDA is not available")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("--device=mps requested but MPS is not available")

    device = torch.device(args.device)

    cfg = TrainConfig(
        nx=args.nx,
        dt=args.dt,
        steps=args.steps,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        noise_level=args.noise,
    )

    train(
        train_path=args.train,
        val_path=args.val,
        ckpt_path=args.ckpt,
        cfg=cfg,
        device=device,
        features=[1, 64, 64, 64, 64, 64, 1],
    )


if __name__ == "__main__":
    main()
