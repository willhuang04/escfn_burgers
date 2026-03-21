# from EntropyStableScheme import KurganovTadmorScheme

import time
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
import argparse
import matplotlib
from absl import logging
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import orbax

from typing import Any, Callable, Sequence
import flax
import optax
from functools import partial

from flax.training import checkpoints, train_state, orbax_utils
from flax import struct, serialization
import orbax.checkpoint as ocp
import shutil
from jax.sharding import Mesh, PartitionSpec
from etils import epath
import matplotlib.pyplot as plt

matplotlib.use('Agg')

import pdb

import jax.numpy as jnp
import jax.lax.linalg as lax_linalg
from jax import custom_jvp
from functools import partial
import jax
from jax import lax
from jax.numpy.linalg import solve
from typing import Any, Callable, Sequence, Tuple
from jax import pure_callback, debug
import numpy as np

jax.config.update("jax_disable_jit",False)
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_enable_x64", False)



class FluxSpectral(nn.Module):
    CFNFeatures: Sequence[int]
    CFN_act: Callable
    SpectralFeatures: Sequence[int]    
    Spectral_act: Callable
    
    def setup(self):
        self.flux_layers = [nn.Dense(feat) for feat in self.CFNFeatures]
        self.flux_out = nn.Dense(6)
        
        self.spectral_layers = [nn.Dense(feat) for feat in self.SpectralFeatures]
        self.spectral_out = nn.Dense(
            1, 
            kernel_init=nn.initializers.zeros_init(),
            bias_init=nn.initializers.zeros_init()
        )
        
        # self.flux_norms = [nn.LayerNorm() for _ in self.CFNFeatures]
        # self.spectral_norms = [nn.LayerNorm() for _ in self.SpectralFeatures]
    
    def forward_flux(self, x):
        for lyr in self.flux_layers:
            x = lyr(x)
            # x = norm(x)
            x = self.CFN_act(x)
        y = self.flux_out(x)
        return y
    
    def forward_spectral(self, x):
        for lyr in self.spectral_layers:
            x = lyr(x)
            # x = norm(x)
            x = self.Spectral_act(x)
        y = nn.softplus(self.spectral_out(x))
        return y
    
    def __call__(self, ConservativeVariables, ):
        """
        ConservativeVariables Shape: (..., VariableSize) e.g., (Batch, X, Y, 3)
        """
        
        batch_shape = ConservativeVariables.shape[:-1]
        u_flat = ConservativeVariables.reshape(-1, ConservativeVariables.shape[-1])
        
        fluxes = jax.vmap(self.forward_flux)(u_flat)
        fluxes_x = fluxes[:, :3]
        fluxes_y = fluxes[:, 3:]

        jac_x = jax.vmap(jax.jacfwd(lambda u: self.forward_flux(u)[:3]))(u_flat)
        jac_y = jax.vmap(jax.jacfwd(lambda u: self.forward_flux(u)[3:]))(u_flat)
        
        jac_x_flat = jac_x.reshape(jac_x.shape[0], -1)
        jac_y_flat = jac_y.reshape(jac_y.shape[0], -1)
        
        combined_jacs = jnp.concatenate([jac_x_flat, jac_y_flat], axis=0)
        combined_rhos = self.forward_spectral(combined_jacs)
        rho_x, rho_y = jnp.split(combined_rhos, 2, axis=0)
        
        fluxes_x = fluxes_x.reshape(*batch_shape, -1)
        fluxes_y = fluxes_y.reshape(*batch_shape, -1)
        rho_x = rho_x.reshape(*batch_shape, 1)
        rho_y = rho_y.reshape(*batch_shape, 1)
        
        return fluxes_x, fluxes_y, rho_x, rho_y


# 4. Different Schemes
class KurganovTadmorScheme(nn.Module):

    def __init__(self, CFNFeatures, CFN_act, SpectralFeatures, Spectral_act, dt, dx, dy, rng = None, boundary="same", limiter="minmod"):
        if rng is None:
            rng = jax.random.PRNGKey(0)
        self.Num_flux = FluxSpectral(CFNFeatures, CFN_act, SpectralFeatures, Spectral_act)
        self.dt = dt
        self.dx = dx
        self.dy = dy
        self.boundary = boundary.lower()
        self.limiter = limiter.lower() 
        self.rng = rng
    
    def flux(self, up, params):
        """
        up: shape (Batch, X, Y, VariableSize)
        """
        fluxes_x, fluxes_y, rho_x, rho_y = self.Num_flux.apply({'params': params['flux']}, up)
        return fluxes_x, fluxes_y, rho_x, rho_y
        
    def Kurganov_Tadmor(self, u, params):
        """
        u: shape (Batch, X, Y, VariableSize)
        """
        u_padded = jnp.pad(u, ((0,0), (2,2), (2,2), (0,0)), mode='wrap')
        
        def get_reconstruction(um, uc, up):
            # Pure vectorized minmod
            a = uc - um
            b = up - uc
            c = (a + b) / 2.0
            min_mag = jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c)))
            slope = jnp.where(a * b > 0, jnp.sign(a) * min_mag, 0.0)
            return uc + 0.5 * slope, uc - 0.5 * slope

        # Reconstruct interfaces in X and Y using vectorized slicing
        uL_x_all, uR_x_all = get_reconstruction(u_padded[:, :-2, 2:-2, :], u_padded[:, 1:-1, 2:-2, :], u_padded[:, 2:, 2:-2, :])
        uL_y_all, uR_y_all = get_reconstruction(u_padded[:, 2:-2, :-2, :], u_padded[:, 2:-2, 1:-1, :], u_padded[:, 2:-2, 2:, :])

        # Interface values (L is left of interface, R is right of interface)
        uL_x = uL_x_all[:, :-1, :, :]
        uR_x = uR_x_all[:, 1:, :, :]
        
        uL_y = uL_y_all[:, :, :-1, :]
        uR_y = uR_y_all[:, :, 1:, :]
        
        fx_L, _, rho_x_L, _ = self.flux(uL_x, params)
        fx_R, _, rho_x_R, _ = self.flux(uR_x, params)
        a_x = jnp.maximum(rho_x_L, rho_x_R)
        
        _, fy_L, _, rho_y_L = self.flux(uL_y, params)
        _, fy_R, _, rho_y_R = self.flux(uR_y, params)
        a_y = jnp.maximum(rho_y_L, rho_y_R)
        
        H_x = 0.5 * (fx_L + fx_R - a_x * (uR_x - uL_x))
        H_y = 0.5 * (fy_L + fy_R - a_y * (uR_y - uL_y))
        Diff_x = lambda H:-(H[1:] - H[:-1])/self.dx
        Diff_y = lambda H:-(H[1:] - H[:-1])/self.dy
        diff_x = jax.vmap(jax.vmap(Diff_x, in_axes=1, out_axes=1), in_axes=0)(H_x)
        diff_y = jax.vmap(jax.vmap(Diff_y, in_axes=0, out_axes=0), in_axes=0)(H_y)
        result = diff_x + diff_y
        return result

    def linearExtrapolation(self, u):
        """
        usize signal_length + 4 x 1
        """
        um = u[:-2]
        u_ = u[1:-1]
        up = u[2:]
        
        def minmod(a, b):
            c = (a + b) / 2.0
            min_mag = jnp.minimum(jnp.abs(a), jnp.minimum(jnp.abs(b), jnp.abs(c)))
            return jnp.where(a * b > 0, jnp.sign(a) * min_mag, 0.0)
        
        uL = u_  + 0.5 * minmod(u_ - um, up - u_) 
        uR = u_  - 0.5 * minmod(u_ - um, up - u_) 
        return uL[:-1], uR[1:]
              
    def TVD_RK3(self, params, u):
        """
        Integrator of Runge Kutta 3 TVD
        """
        
        u1 = u + self.dt * self.Kurganov_Tadmor(u, params)
        u2 = 3/4 * u + 1/4 * u1 + 1/4 * self.dt * self.Kurganov_Tadmor(u1, params)
        u3 = 1/3 * u + 2/3 * u2 + 2/3 * self.dt * self.Kurganov_Tadmor(u2, params)
        return u3
    
    def euler(self, params, u):
        """
        Integrator of Euler TVD
        """
        
        u1 = u + self.dt * self.Kurganov_Tadmor(u, params)
        return u1   

@jax.jit
def apply_model(state, un, u_np1):
    """Computes gradients, loss for a single batch."""
    def loss_fn(params):
        # Swap axes so Time is index 0: shape becomes (Time, Batch, X, Y, Var)
        u_targets = jnp.swapaxes(u_np1, 0, 1)
        
        model_apply = state.apply_fn
        
        @jax.remat
        def step_fn(carry_state, target_step):
            um = carry_state
            u = model_apply(params, um)
            step_loss = jnp.mean((target_step - u)**2)
            return u, step_loss
        
        _, step_losses = jax.lax.scan(step_fn, un, u_targets)
        total_loss = jnp.sum(step_losses)
        
        return total_loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss

@jax.jit
def apply_model_val(state, un, u_np1):
    """Computes gradients, loss for a single batch."""

    def loss_fn(params):
        # Swap axes so Time is index 0: shape becomes (Time, Batch, X, Y, Var)
        u_targets = jnp.swapaxes(u_np1, 0, 1)
        
        def step_fn(carry_state, target_step):
            um = carry_state
            u = state.apply_fn(params, um)
            step_loss = jnp.mean((target_step - u)**2)
            return u, step_loss
            
        _, step_losses = jax.lax.scan(step_fn, un, u_targets)
        total_loss = jnp.sum(step_losses)
        
        return total_loss

    return loss_fn(state.params)

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)

@partial(jax.jit, static_argnames=['batch_size'])
def train_epoch(state, train_ds, val_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = train_ds['un'].shape[0]
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, len(train_ds['un']))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    
    def step_fn(current_state, perm):
        batch_input = train_ds['un'][perm, ...]
        batch_output = train_ds['un_p1'][perm, ...]
        grads, loss = apply_model(current_state, batch_input, batch_output)
        next_state = update_model(current_state, grads)
        
        return next_state, loss 

    final_state, epoch_loss = jax.lax.scan(step_fn, state, perms)
    train_loss = jnp.mean(epoch_loss)
    # Validate
    val_loss = apply_model_val(final_state, val_ds['un'], val_ds['un_p1'])
    return final_state, train_loss, val_loss


def get_Datasets(IC_Noise_level, Noise_level, rng=None, L = 10, data_path = 'Data/trainData.npy'):
    """Load Dataset"""
    if rng is None:
        rng = jax.random.PRNGKey(100)
    trainData = np.load(data_path)
    train_ds = {}    
    un = []
    un_p1 = []
    for i in range(trainData.shape[0]):
        rng, sample_rng = jax.random.split(rng)
        index = 0 # jax.random.choice(sample_rng, 20-L)
        un.append(trainData[i:i+1,index,:,:,:])
        un_p1.append(trainData[i:i+1, index+1:index+L+1,:,:,:]) 
    un = np.concatenate(un, axis=0)
    un_p1 = np.concatenate(un_p1, axis=0)
    axis_to_average = tuple(range(trainData.ndim - 1)) 
    dataset_mean_abs = np.mean(np.abs(trainData), axis=axis_to_average, keepdims=True)
    dataset_mean_abs = np.squeeze(dataset_mean_abs)
    rng1, rng = jax.random.split(rng)
    un += jax.random.normal(rng, shape=un.shape) * dataset_mean_abs * IC_Noise_level
    shape = un_p1.shape
    un_p1 += jax.random.normal(rng1, shape=shape) * dataset_mean_abs * Noise_level
    train_ds['un'] = un # un.reshape(-1, un.shape[2],1)
    train_ds['un_p1'] = un_p1 # un_p1.reshape(-1, un_p1.shape[2], 1)
    return train_ds  



def create_train_state(ess, rng, learning_rate):
    """Creates initial `TrainState`."""
    rng1, rng2 = jax.random.split(rng)
    params1 = ess.Num_flux.init(rng1, jnp.ones((3,)))['params']
    params = {'flux': params1}
    tx = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(learning_rate, weight_decay=1e-4))    
    return train_state.TrainState.create(apply_fn=ess.TVD_RK3, params=params, tx=tx)


def TrainEntropyStableScheme(nx, ny, dt, epochs, batch_size, lr, ic_noise, noise, timesteps, resume_training, ckpt_dir='./ckpts/Edge/'):
    
    Nx = nx
    Ny = ny
    dx = 1/Nx
    dy = 1/Ny
    dt = dt
    batch_size = batch_size
    start_epoch = 1
    rng = jax.random.PRNGKey(0)
    vector_rng, rng = jax.random.split(rng)
    EntropyStableForm = KurganovTadmorScheme(rng = vector_rng, CFNFeatures=[16, 16], CFN_act=nn.silu, SpectralFeatures=[8], Spectral_act=nn.silu, dt=dt, dx=dx, dy=dy, boundary="same", limiter="minmod")
    
    rng, gendata_rng = jax.random.split(rng)
    timeSteps = timesteps
    Noise_level = noise
    IC_Noise_level = ic_noise
    train_ds = get_Datasets(IC_Noise_level, Noise_level, L=timeSteps, rng = gendata_rng, data_path = 'Data/trainData_2D_shallow_water_'+str(Nx)+'.npy')
    rng, gendata_rng = jax.random.split(rng)
    val_ds = get_Datasets(IC_Noise_level, Noise_level, L=timeSteps, rng = gendata_rng, data_path = 'Data/valData_2D_shallow_water_' + str(Nx) + '.npy')
    #schedule = optax.piecewise_constant_schedule(lr, boundaries_and_scales={4000:0.3, 10000:0.1})
                                                 #{2000:1/2, 4000:1/2, 8000:1/2, 10000:1/2,15000:1/2})  
    # schedule = optax.exponential_decay(lr, 2000, 0.95, end_value=1e-6)
    total_steps = (len(train_ds['un']) // batch_size) * epochs
    schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps, alpha=1)
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(EntropyStableForm, init_rng, schedule)
    #orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        max_to_keep=3, 
        create=True
    )
    
    path = epath.Path(ckpt_dir + 'orbax/single_save')
    ckpt_dir_absolute = os.path.abspath(path)
    path = epath.Path(ckpt_dir_absolute)
    mngr = ocp.CheckpointManager(
        path, 
        options=options
    )
    if path.exists() and resume_training:            
        target = state
        latest_step = mngr.latest_step()
        state_restored = mngr.restore(latest_step, args=ocp.args.StandardRestore(target))
        state = state_restored
        start_epoch = latest_step + 1
        print(f"Successfully resumed from epoch: {latest_step}. Next epoch will be {start_epoch}.")
    else:
        if path.exists():
            shutil.rmtree(path)
        path.mkdir()
        mngr = ocp.CheckpointManager(
            path, 
            options=options
        )   

    best_val_loss = float('inf')  # Initialize the best validation loss to infinity

    for epoch in range(start_epoch, epochs + 1):
        rng, input_rng = jax.random.split(rng)
        state, train_loss, val_loss = train_epoch(
            state, train_ds, val_ds,  batch_size, input_rng
        )

        print(
            'epoch:% 3d, train_loss: %.10f, val_loss: %.10f' % (
                epoch,
                train_loss,
                val_loss
            )
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f'New best validation loss: {val_loss:.10f}, saving model...')
            ckpt = {'model': state}
            mngr.save(epoch, args=ocp.args.StandardSave(state))
            mngr.wait_until_finished()
#         if epoch % 1 == 0:
#             evaluateESS(epoch, ckpt_dir=ckpt_dir)
    
def state_shape(state):
    return jax.eval_shape(lambda: state)

def evaluateESS(nx, ny, dt, eval_steps, ckpt_dir='ckpts/Edge/'):
    
    Nx = nx
    Ny = ny
    dx = 1/Nx
    dy = 1/Ny
    dt = dt
    lr = 1e-3
    # batch_size = 4
    rng = jax.random.PRNGKey(100)
    vector_rng, rng = jax.random.split(rng)    
    EntropyStableForm = KurganovTadmorScheme(rng = vector_rng, CFNFeatures=[16, 16], CFN_act=nn.silu, SpectralFeatures=[8], Spectral_act=nn.silu, dt=dt, dx=dx, dy=dy, boundary="same", limiter="minmod")
    rng, init_rng = jax.random.split(rng)
    schedule = optax.exponential_decay(lr, 2000, 0.95, end_value=1e-6)
    state = create_train_state(EntropyStableForm, init_rng, schedule)
    path = epath.Path(ckpt_dir + 'orbax/single_save')
    ckpt_dir_absolute = os.path.abspath(path)
    path = epath.Path(ckpt_dir_absolute)
    mngr = ocp.CheckpointManager(path)
    
    try:
        state = mngr.restore(
            mngr.latest_step(), 
            args=ocp.args.StandardRestore(state)
        )
        print("Checkpoint restored successfully.")
    except ValueError as e:
        print(f"Restore failed: {e}")
        return
    
    dir_path = ckpt_dir.split("/")
    plot_folder = dir_path[2]
    if not os.path.exists('_plots/' + plot_folder):
        os.mkdir('_plots/' + plot_folder)
    if not os.path.exists('_plots/' + plot_folder + '/Entropy'):
        os.mkdir('_plots/' + plot_folder + '/Entropy')   
    if not os.path.exists('_plots/' + plot_folder + '/Solutions'):
        os.mkdir('_plots/' + plot_folder + '/Solutions')   
    if not os.path.exists('_plots/' + plot_folder + '/Conserved_quantities'):
        os.mkdir('_plots/' + plot_folder + '/Conserved_quantities') 
    
    data_path =  'Data/testData_2D_shallow_water_'+str(Nx) +'.npy'
    testData = np.load(data_path)

    N = eval_steps
    x_start = 0
    x_end = 1
    y_start = 0
    y_end = 1
    t = np.linspace(0.0, N * dt, N + 1)
    
    un = testData[:, 0, :, :, :]
    un_p1 = testData[:, :, :, :, :]
    
    @jax.jit
    def rollout(start_state):
        def step_fn(current_u, _):
            next_u = state.apply_fn(state.params, current_u)
            return next_u, next_u
        _, u_pred_steps = jax.lax.scan(step_fn, start_state, xs=None, length=N)
        return u_pred_steps
    
    u_pred_steps = rollout(un)
    un_expanded = jnp.expand_dims(un, axis=0) 
    u_pred_stacked = jnp.concatenate([un_expanded, u_pred_steps], axis=0)
    u_pred = jnp.swapaxes(u_pred_stacked, 0, 1)
    
    no_traj = testData.shape[0]
    avg_error = 0
    for i in range(no_traj):
        exact_state = testData[i, N, :, :, 0]
        pred_state = u_pred[i, N, :, :, 0]
        l2_diff = np.linalg.norm(exact_state - pred_state)
        l2_exact = np.linalg.norm(exact_state) + 1e-8
        avg_error += (l2_diff / l2_exact) / no_traj
    print(f'Average test error: {avg_error:.6f}')
    
    # Snapshot plot at final time.
    pred = u_pred[0, N, :, :, 0]
    exact = un_p1[0, N, :, :, 0] if un_p1.shape[1] > N else None
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
    axes[0].set_title(f"Prediction (t={N*dt:.2f})")
    fig.colorbar(im0, ax=axes[0])
    if exact is not None:
        im1 = axes[1].imshow(exact.T, interpolation='nearest', cmap='jet', 
                            origin='lower', extent=[x_start, x_end, y_start, y_end], vmin=vmin, vmax=vmax)
        axes[1].set_title("Exact")
        fig.colorbar(im1, ax=axes[1])
        err = np.abs(pred - exact) / (np.abs(exact) + 1e-8)
        im2 = axes[2].imshow(err.T, interpolation='nearest', cmap='binary', 
                            origin='lower', extent=[x_start, x_end, y_start, y_end],) 
        axes[2].set_title("Rel Error")
        fig.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig("_plots/" + plot_folder + "/Solutions/" + str(N).zfill(3) + ".png",dpi=300)
    plt.close()
    
    grav = 1.0
    def get_swe_entropy(data, step):
        h = data[0, step, :, :, 0]
        u = data[0, step, :, :, 1] / (h + 1e-8)
        v = data[0, step, :, :, 2] / (h + 1e-8)
        entropy = 0.5 * h * (u ** 2 + v ** 2) + 0.5 * grav * h ** 2
        return float(np.mean(entropy))

    # Entropy: h*(u^2+v^2)/2+g*h^2/2
    if un_p1.shape[1] >= N + 1:
        exact_entropy = [get_swe_entropy(un_p1, s) for s in range(N+1)]
    else:
        exact_entropy = None
    pred_entropy = [get_swe_entropy(u_pred, s) for s in range(N + 1)]

    plt.figure(figsize=(4, 3))
    if exact_entropy is not None:
        plt.plot(t, exact_entropy, label="Exact")
    plt.plot(t, pred_entropy, "-.", label="Pred")
    plt.xlabel("t / s")
    plt.ylabel("Entropy")
    plt.legend()
    plt.tight_layout()
    plt.savefig("_plots/" + plot_folder + "/Entropy/entropy.png",dpi=300)
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
    if un_p1.shape[1] >= N + 1:
        exact_m0, exact_px0, exact_py0 = get_conserved_quantities(un_p1, 0)
        exact_m_err, exact_px_err, exact_py_err = [], [], []
        for s in range(N+1):
            exact_ms, exact_pxs, exact_pys = get_conserved_quantities(un_p1, s)
            exact_m_err.append(exact_ms - exact_m0)
            exact_px_err.append(exact_pxs - exact_px0)
            exact_py_err.append(exact_pys - exact_py0)
    else:
        exact_m_err, exact_px_err, exact_py_err = None, None, None

    pred_m0, pred_px0, pred_py0 = get_conserved_quantities(u_pred, 0)
    pred_m_err, pred_px_err, pred_py_err = [], [], []
    for s in range(N + 1):
        pred_ms, pred_pxs, pred_pys = get_conserved_quantities(u_pred, s)
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
    plt.savefig("_plots/" + plot_folder + "/Conserved_quantities/conserved_mass.png",dpi=300)
    plt.close()
    
    plt.figure(figsize=(4, 3))
    if exact_px_err is not None:
        plt.plot(t, exact_px_err, label="Exact")
    plt.plot(t, pred_px_err, "-.", label="Pred")
    plt.xlabel("t / s")
    plt.ylabel("Conserved_x_momentum")
    plt.legend()
    plt.tight_layout()
    plt.savefig("_plots/" + plot_folder + "/Conserved_quantities/conserved_x_momentum.png",dpi=300)
    plt.close()
    
    plt.figure(figsize=(4, 3))
    if exact_py_err is not None:
        plt.plot(t, exact_py_err, label="Exact")
    plt.plot(t, pred_py_err, "-.", label="Pred")
    plt.xlabel("t / s")
    plt.ylabel("Conserved_y_momentum")
    plt.legend()
    plt.tight_layout()
    plt.savefig("_plots/" + plot_folder + "/Conserved_quantities/conserved_y_momentum.png",dpi=300)
    plt.close()
    
if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--mode", choices=["train", "check", "eval"], default="train")
    parser.add_argument("--resume", action="store_true", help="Flag to resume from the latest checkpoint")
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--dt", type=float, default=0.000625)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--ic_noise", type=float, default=0.0)
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--eval_nx", type=int, default=256)
    parser.add_argument("--eval_ny", type=int, default=256)
    parser.add_argument("--eval_dt", type=float, default=0.000625)
    parser.add_argument("--eval_steps", type=int, default=20)
    
    args = parser.parse_args()
    
    print('Starting...')
    start_time = time.perf_counter()
        
    my_folder = './ckpts/KT_DNN_train' + str(args.nx) + '_test' + str(args.eval_nx) + '/'
    
    if args.mode == "train":
        TrainEntropyStableScheme(args.nx, args.ny, args.dt, args.epochs, args.batch_size, args.lr, args.ic_noise, args.noise, args.steps, args.resume, ckpt_dir=my_folder)
    
    if args.mode in ["train", "eval", "check"]:
        evaluateESS(args.eval_nx, args.eval_ny, args.eval_dt, args.eval_steps, ckpt_dir=my_folder)
    
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f'Runtime: {elapsed_time:.6f} seconds')