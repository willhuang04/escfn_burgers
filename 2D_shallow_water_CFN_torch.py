import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import vmap, jacrev, functional_call
from torch.utils.data import DataLoader, Dataset

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class FluxAndSpectralNet(nn.Module):
    def __init__(self, features: List[int], hidden_dim: int):
        super().__init__()
        if len(features) < 2:
            raise ValueError("features must include input and output sizes, e.g. [64, ..., 1]")
        layers: List[nn.Module] = []
        for in_f, out_f in zip(features[:-1], features[1:]):
            layers.append(nn.Linear(in_f, out_f))
            if out_f != features[-1]:
                layers.append(nn.SiLU())
        self.flux_net = nn.Sequential(*layers)
        
        n_inputs = features[0]
        n_flux_outputs = features[-1]
        jac_dim = n_inputs * (n_flux_outputs // 2) 
        
        self.spectral_net = nn.Sequential(
            nn.Linear(jac_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # 1. Flatten spatial dims for processing: (N_total, 3)
        batch_shape = u.shape[:-1]
        u_flat = u.reshape(-1, u.shape[-1])
        
        # 2. Compute Fluxes normally
        fluxes = self.flux_net(u_flat)
        fluxes_x = fluxes[:, :3]
        fluxes_y = fluxes[:, 3:]
        
        # 3. Compute Jacobians efficiently using torch.func
        # We need J_x = d(Flux_X)/du and J_y = d(Flux_Y)/du
        jac_x, jac_y = self._compute_jacobians(u_flat)
        
        # 4. Predict Wave Speeds using the Spectral Net
        # Flatten Jacobians: (B, 3, 3) -> (B, 9)
        rho_x = torch.abs(self.spectral_net(jac_x.flatten(start_dim=1)))
        rho_y = torch.abs(self.spectral_net(jac_y.flatten(start_dim=1)))
        
        # 5. Reshape back to grid
        fluxes_x = fluxes_x.reshape(*batch_shape, -1)
        fluxes_y = fluxes_y.reshape(*batch_shape, -1)
        rho_x = rho_x.reshape(*batch_shape, 1)
        rho_y = rho_y.reshape(*batch_shape, 1)
        
        return fluxes_x, fluxes_y, rho_x, rho_y

    def _compute_jacobians(self, u_flat):
        """
        Computes per-sample Jacobian matrices using vmap + jacrev.
        This is much faster than looping.
        """
        
        # Define a pure function for the X-flux (first 3 outputs)
        def get_fx(params, x):
            out = functional_call(self.flux_net, params, (x.unsqueeze(0),))
            return out.squeeze(0)[:3] # Take first 3 channels (F)

        # Define a pure function for the Y-flux (last 3 outputs)
        def get_fy(params, x):
            out = functional_call(self.flux_net, params, (x.unsqueeze(0),))
            return out.squeeze(0)[3:] # Take last 3 channels (G)

        params = dict(self.flux_net.named_parameters())
        
        chunk = 1024
        
        J_x = vmap(jacrev(get_fx, argnums=1), in_dims=(None, 0), chunk_size=chunk)(params, u_flat)
        J_y = vmap(jacrev(get_fy, argnums=1), in_dims=(None, 0), chunk_size=chunk)(params, u_flat)
        
        return J_x, J_y
        
class KurganovTadmorSchemeTorch(nn.Module):
    def __init__(
        self,
        features: List[int],
        hidden_dim: int, 
        dt: float,
        dx: float,
        dy: float,
        boundary: str = "same",
        limiter: str = "minmod",
    ):
        super().__init__()
        self.num_flux = FluxAndSpectralNet(features, hidden_dim)
        self.dt = float(dt)
        self.dx = float(dx)
        self.dy = float(dy)
        self.boundary = boundary.lower()
        self.limiter = limiter.lower()

    def _minmod(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Matches the JAX implementation in ktBurgers_CFN.py
        zeros = torch.zeros_like(a)
        # The limiter returns 0 if the slopes are in opposite directions (extrema)
        same_sign = (torch.sign(a) == torch.sign(b))
        
        # Calculate the three bounds from the paper's min() function
        abs_a = torch.abs(a)
        abs_b = torch.abs(b)
        abs_avg = torch.abs(a + b) / 2.0
        
        # Find the minimum of the three
        min_mag = torch.minimum(torch.minimum(abs_a, abs_b), abs_avg)
        
        # Apply the sign if same_sign is True, otherwise return 0
        return torch.where(same_sign, torch.sign(a) * min_mag, zeros)

    def linear_extrapolation(self, u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        u: (B, Nx+4, Ny+4, C)
        returns:
          uL_x, uR_x: (B, Nx+1, Ny, C)
          uL_y, uR_y: (B, Nx, Ny+1, C)
        """
        um_x = u[:, :-2, :, :]
        u_x = u[:, 1:-1, :, :]
        up_x = u[:, 2:, :, :]

        slope_x = self._minmod(u_x - um_x, up_x - u_x)
        uL_x = u_x + 0.5 * slope_x
        uR_x = u_x - 0.5 * slope_x
        
        um_y = u[:, :, :-2, :]
        u_y = u[:, :, 1:-1, :]
        up_y = u[:, :, 2:, :]

        slope_y = self._minmod(u_y - um_y, up_y - u_y)
        uL_y = u_y + 0.5 * slope_y
        uR_y = u_y - 0.5 * slope_y
        
        return uL_x[:, :-1, :, :], uR_x[:, 1:, :, :], uL_y[:, :, :-1, :], uR_y[:, :, 1:, :]

    def flux_and_dflux(self, up: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        up: (B, Npx, Npy, C)
        returns:
          f(up): (B, Npx, Npy, C)
          df/dup: (B, Npx, Npy, C)
        """
        # Validation/eval often wraps forward passes in torch.no_grad(), but we still need
        # first-order gradients w.r.t. inputs to estimate wave speeds.
        with torch.enable_grad():
            up_leaf = up.detach().requires_grad_(True)
            b, nx, ny, c = up_leaf.shape
            f_x, f_y, rho_x, rho_y = self.num_flux(up_leaf.reshape(b * nx * ny, c))
            
        return f_x, f_y, rho_x, rho_y

    def kurganov_tadmor(self, u_padded: torch.Tensor) -> torch.Tensor:
        """
        u_padded: (B, Nx+4, Ny+4, C)
        returns rhs: (B, Nx, Ny, C)
        """
        uL_x, uR_x, uL_y, uR_y = self.linear_extrapolation(u_padded)
        fx_L, _, rho_x_L, _ = self.num_flux(uL_x)
        fx_R, _, rho_x_R, _ = self.num_flux(uR_x)

        a_x = torch.maximum(rho_x_L, rho_x_R)
        a_x = torch.clamp(a_x, min=1.0)

        # Kurganov-Tadmor Numerical Flux Hx
        Hx = 0.5 * (fx_L + fx_R - a_x * (uR_x - uL_x))
        
        _, fy_L, _, rho_y_L = self.num_flux(uL_y)
        _, fy_R, _, rho_y_R = self.num_flux(uR_y)

        a_y = torch.maximum(rho_y_L, rho_y_R)
        a_y = torch.clamp(a_y, min=1.0)

        # Kurganov-Tadmor Numerical Flux Hy
        Hy = 0.5 * (fy_L + fy_R - a_y * (uR_y - uL_y))
        
        diff_x = (Hx[:, 1:, :, :] - Hx[:, :-1, :, :]) / self.dx
        diff_y = (Hy[:, :, 1:, :] - Hy[:, :, :-1, :]) / self.dy
        
        diff_x_interior = diff_x[:, :, 2:-2, :]
        diff_y_interior = diff_y[:, 2:-2, :, :]
        
        res = -(diff_x_interior + diff_y_interior)

        return res

    def rhs(self, u: torch.Tensor) -> torch.Tensor:
        # Periodic wrap padding by 2 cells on each side (matches jnp.pad(..., mode="wrap")).
        if self.boundary != "same":
            raise NotImplementedError(f"Only boundary='same' (periodic wrap) supported, got {self.boundary!r}")
        u_padded = F.pad(u, (0, 0, 2, 2, 2, 2), mode='circular')
        return self.kurganov_tadmor(u_padded)

    def tvd_rk3(self, u: torch.Tensor) -> torch.Tensor:
        u1 = u + self.dt * self.rhs(u)
        u2 = 0.75 * u + 0.25 * u1 + 0.25 * self.dt * self.rhs(u1)
        u3 = (1.0 / 3.0) * u + (2.0 / 3.0) * u2 + (2.0 / 3.0) * self.dt * self.rhs(u2)
        return u3

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.tvd_rk3(u)


class ShallowWaterRolloutDataset(Dataset):
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
        self.arr = np.load(data_path, mmap_mode='r')
        if self.arr.ndim != 5:
            raise ValueError(f"Expected data array with ndim=5 (S,T,Nx,Ny,C), got {self.arr.shape}")
        if index + rollout_steps >= self.arr.shape[1]:
            raise ValueError(
                f"rollout_steps={rollout_steps} too large for T={self.arr.shape[1]} at index={index}"
            )
        
        self.rollout_steps = rollout_steps
        self.noise_level = noise_level
        self.index = index
        self.dtype = dtype
        
        # Setup RNG for dynamic noise
        self.rng = np.random.default_rng(seed)
        
        # Calculate scale once (using a small slice to avoid loading everything)
        # We assume the stats of the first batch are representative of the whole
        sample_slice = self.arr[0:10] 
        self.scale = float(np.mean(np.abs(sample_slice))) * float(noise_level)

    def __len__(self) -> int:
        return self.arr.shape[0]

    def __getitem__(self, idx: int):
        # 1. Load the specific data for this index
        # Shape: (Nx, Ny, C)
        un = self.arr[idx, self.index].astype(self.dtype)
        
        # Shape: (Rollout, Nx, Ny, C)
        un_p1 = self.arr[idx, self.index + 1 : self.index + 1 + self.rollout_steps].astype(self.dtype)

        # 2. Add Dynamic Noise 
        if self.noise_level > 0:
            # Generate noise that matches 'un' (3D: Nx, Ny, C)
            noise = self.rng.normal(size=un.shape).astype(self.dtype) * self.scale
            un = un + noise

        # 3. Return Tensors
        return torch.from_numpy(un), torch.from_numpy(un_p1)


@dataclass
class TrainConfig:
    nx: int = 256
    ny: int = 256
    dt: float = 0.005
    steps: int = 20
    batch_size: int = 2
    epochs: int = 40
    lr: float = 1e-3
    noise_level: float = 0.0
    decay_steps: int = 2000
    decay_rate: float = 0.85
    end_lr: float = 1e-6

# This is still the loss in the original paper
def multistep_rollout_loss(model: nn.Module, un: torch.Tensor, u_np1: torch.Tensor, entropy_weight=10.0) -> torch.Tensor:
    """
    un: (B, Nx, Ny, C)
    u_np1: (B, L, Nx, Ny, C)
    """
    um = un
    total = 0.0
    for i in range(u_np1.shape[1]):
        u_pred = model(um)
        total = total + F.mse_loss(u_pred, u_np1[:, i, :, :, :])
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
    hidden_dim: int,
    features: Optional[List[int]] = None,
) -> nn.Module:
    if features is None:
        # Input channel count is 3 for 2D shallow water here.
        # Output channel count is 6 for 2 fluxes each depending on 3 variables.
        features = [3, 128, 128, 128, 128, 128, 6]
    dx = 1 / cfg.nx
    dy = 1 / cfg.ny

    train_ds = ShallowWaterRolloutDataset(train_path, rollout_steps=cfg.steps, noise_level=cfg.noise_level, seed=0)
    val_ds = ShallowWaterRolloutDataset(val_path, rollout_steps=cfg.steps, noise_level=cfg.noise_level, seed=1)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = KurganovTadmorSchemeTorch(hidden_dim=hidden_dim, features=features, dt=cfg.dt, dx=dx, dy=dy).to(device)
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

            # Add batch dimension shape: (B, Nx, Ny, C) and (B, L, Nx, Ny, C)
            if un.ndim == 3:
                un = un.unsqueeze(0)
            if un_p1.ndim == 4:
                un_p1 = un_p1.unsqueeze(0)

            lr = lr_schedule(global_step, cfg)
            set_optimizer_lr(optimizer, lr)

            optimizer.zero_grad(set_to_none=True)
            loss = multistep_rollout_loss(model, un, un_p1)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().cpu().item())
            torch.mps.empty_cache()

            train_losses.append(loss.detach().cpu().item())
            global_step += 1

        model.eval()
        with torch.no_grad():
            val_losses = []
            for un, un_p1 in val_loader:
                un = un.to(device=device, dtype=torch.float32)
                un_p1 = un_p1.to(device=device, dtype=torch.float32)
                if un.ndim == 3:
                    un = un.unsqueeze(0)
                if un_p1.ndim == 4:
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
                    "hidden_dim": hidden_dim,
                },
                ckpt_path,
            )

    return model


def load_model(ckpt_path: str, device: torch.device, dt: float, dx: float, dy: float) -> KurganovTadmorSchemeTorch:
    ckpt = torch.load(ckpt_path, map_location=device)
    features = ckpt.get("features")
    hidden_dim = ckpt.get("hidden_dim")
    if features is None:
        raise ValueError("Checkpoint missing 'features'")
    model = KurganovTadmorSchemeTorch(features=features, hidden_dim=hidden_dim, dt=dt, dx=dx, dy=dy).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def evaluate(
    ckpt_path: str,
    test_path: str,
    out_dir: str,
    nx: int,
    ny: int, 
    dt: float,
    device: torch.device,
    n_steps: int = 600,
) -> None:
    """Rough PyTorch equivalent of evaluateESS: rollout and save a few plots."""
    import matplotlib.pyplot as plt

    dx = 1 / nx
    dy = 1 / ny
    model = load_model(ckpt_path, device=device, dt=dt, dx=dx, dy=dy)

    test_data = np.load(test_path, mmap_mode='r')
    if test_data.ndim != 5:
        raise ValueError(f"Expected test data with ndim=5 (S,T,Nx,Ny,C), got {test_data.shape}")
    if test_data.shape[2] != nx:
        raise ValueError(f"nx mismatch: expected N={nx}, but test data has N={test_data.shape[2]}")
    if test_data.shape[3] != ny:
        raise ValueError(f"ny mismatch: expected N={ny}, but test data has N={test_data.shape[3]}")

    # Use the first trajectory.
    un0 = torch.from_numpy(test_data[:1, 0, :, :, :].copy()).to(device=device, dtype=torch.float32)

    u_rollout = [un0]
    
    with torch.no_grad():
        un = un0
        for i in range(n_steps):
            un = model(un)
            # THE KILL SWITCH
            if torch.isnan(un).any() or torch.isinf(un).any():
                print(f"💥 Instability detected! Model blew up at step {i}.")
                n_steps = i  
                break
            u_rollout.append(un)

    x_start = 0
    x_end = 1
    y_start = 0
    y_end = 1
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)

    u_np = torch.stack(u_rollout, dim=1).detach().cpu().numpy()  # (1, n_steps+1, Nx, Ny, C)

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
    pred = u_np[0, j, :, :, 0]
    exact = test_data[0, j, :, :, 0] if test_data.shape[1] > j else None
    if exact is not None:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        # Determine common color scale for fair comparison
        vmin = min(np.min(exact), np.min(pred))
        vmax = max(np.max(exact), np.max(pred))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(4, 3.5))
        axes = [axes] # Wrap in list to make iterable
        vmin, vmax = np.min(pred), np.max(pred)    
    im0 = axes[0].imshow(pred.T, interpolation='nearest', cmap='jet', 
                        origin='lower', extent=[x_start, x_end, y_start, y_end], vmin=vmin, vmax=vmax)
    axes[0].set_title(f"Prediction (t={j*dt:.2f})")
    fig.colorbar(im0, ax=axes[0])
    if exact is not None:
        im1 = axes[1].imshow(exact.T, interpolation='nearest', cmap='jet', 
                            origin='lower', extent=[x_start, x_end, y_start, y_end], vmin=vmin, vmax=vmax)
        axes[1].set_title("Exact")
        fig.colorbar(im1, ax=axes[1])
        err = np.abs(pred - exact)
        im2 = axes[2].imshow(err.T, interpolation='nearest', cmap='binary', 
                            origin='lower', extent=[x_start, x_end, y_start, y_end],) 
        axes[2].set_title("Abs Error")
        fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(u_dir, f"{j:03d}.png"), dpi=200)
    plt.close()
    
    grav = 1.0
    def get_swe_entropy(data, step):
        h = data[0, step, :, :, 0]
        u = data[0, step, :, :, 1] / (h + 1e-8)
        v = data[0, step, :, :, 2] / (h + 1e-8)
        entropy = 0.5 * h * (u ** 2 + v ** 2) + 0.5 * grav * h ** 2
        return float(np.mean(entropy))

    # Entropy: h*(u^2+v^2)/2+g*h^2/2
    if test_data.shape[1] >= n_steps + 1:
        exact_entropy = [get_swe_entropy(test_data, s) for s in range(n_steps+1)]
    else:
        exact_entropy = None
    pred_entropy = [get_swe_entropy(u_np, s) for s in range(n_steps + 1)]

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
    
    cell_area = dx * dy  # Area element for 2D integration

    def get_conserved_quantities(data, step):
        # Extract variables
        h  = data[0, step, :, :, 0]
        hu = data[0, step, :, :, 1] 
        hv = data[0, step, :, :, 2] 

        # Integrate over domain
        mass = np.sum(h) * cell_area
        mom_x = np.sum(hu) * cell_area
        mom_y = np.sum(hv) * cell_area
        
        return mass, mom_x, mom_y

    # Conserved quantities
    if test_data.shape[1] >= n_steps + 1:
        exact_m0, exact_px0, exact_py0 = get_conserved_quantities(test_data, 0)
        exact_m_err, exact_px_err, exact_py_err = [], [], []
        for s in range(n_steps+1):
            exact_ms, exact_pxs, exact_pys = get_conserved_quantities(test_data, s)
            exact_m_err.append(exact_ms - exact_m0)
            exact_px_err.append(exact_pxs - exact_px0)
            exact_py_err.append(exact_pys - exact_py0)
    else:
        exact_m_err, exact_px_err, exact_py_err = None, None, None

    pred_m0, pred_px0, pred_py0 = get_conserved_quantities(u_np, 0)
    pred_m_err, pred_px_err, pred_py_err = [], [], []
    for s in range(n_steps + 1):
        pred_ms, pred_pxs, pred_pys = get_conserved_quantities(u_np, s)
        pred_m_err.append(pred_ms - pred_m0)
        pred_px_err.append(pred_pxs - pred_px0)
        pred_py_err.append(pred_pys - pred_py0)
            
    plt.figure(figsize=(4, 3))
    if exact_m_err is not None:
        plt.plot(t, exact_m_err, label="Exact")
    plt.plot(t, pred_m_err, "-.", label="Pred")
    plt.xlabel("t / s")
    plt.ylabel("Conserved_mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cons_dir, "conserved_mass.png"), dpi=200)
    plt.close()
    
    plt.figure(figsize=(4, 3))
    if exact_px_err is not None:
        plt.plot(t, exact_px_err, label="Exact")
    plt.plot(t, pred_px_err, "-.", label="Pred")
    plt.xlabel("t / s")
    plt.ylabel("Conserved_x_momentum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cons_dir, "conserved_x_momentum.png"), dpi=200)
    plt.close()
    
    plt.figure(figsize=(4, 3))
    if exact_py_err is not None:
        plt.plot(t, exact_py_err, label="Exact")
    plt.plot(t, pred_py_err, "-.", label="Pred")
    plt.xlabel("t / s")
    plt.ylabel("Conserved_y_momentum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(cons_dir, "conserved_y_momentum.png"), dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "check", "eval"], default="train")
    parser.add_argument("--train", default="Data/trainData_2D_shallow_water_64.npy")
    parser.add_argument("--val", default="Data/valData_2D_shallow_water_64.npy")
    parser.add_argument("--test", default="Data/testData_2D_shallow_water_256.npy")
    parser.add_argument("--ckpt", default="ckpts/kt_torch_best.pt")
    parser.add_argument("--out", default="_plots/kt_torch")
    parser.add_argument("--eval_steps", type=int, default=600)

    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.005)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--noise", type=float, default=0.0)

    parser.add_argument("--device", default="mps", choices=["cpu", "cuda", "mps"])

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
            ny=args.ny,
            dt=args.dt / 10,
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
        nx=64,
        ny=64,
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
        hidden_dim=64,
        features=[3, 128, 128, 128, 128, 128, 6],
    )


if __name__ == "__main__":
    print('Starting...')
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f'Runtime: {elapsed_time:.6f} seconds')

