# from EntropyStableScheme import KurganovTadmorScheme

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

import os

import matplotlib.pyplot as plt

import pdb

import jax.numpy as jnp
import jax.lax.linalg as lax_linalg
from jax import custom_jvp
from functools import partial
import jax
from jax import lax
from jax.numpy.linalg import solve
from typing import Any, Callable, Sequence, Tuple
from jax.experimental import host_callback
import numpy as np

jax.config.update("jax_disable_jit",False)
jax.config.update("jax_debug_nans", True)



class Flux(nn.Module):
    Features: Sequence[int]
    act: Callable    
    def setup(self):
        self.layers = [nn.Dense(feat) for feat in self.Features]
    
    def __call__(self, ConservativeVariables, ):
        """
        ConservativeVariables.Shape: SignalLength x VariableSize 
        
        """
        
        x = ConservativeVariables
        for i, lyr in enumerate(self.layers[:-1]):
            x = self.act(lyr(x))
        x = self.layers[-1](x)
        return x     


# 4. Different Schemes
class KurganovTadmorScheme(nn.Module):

    def __init__(self,rng = jax.random.PRNGKey(0), Features=[10,1], dt=0.001, dx=0.001, boundary="same", limiter="minmod"):

        self.Num_flux = Flux(Features, nn.silu)
        self.dt = dt
        self.dx = dx
        self.boundary = boundary.lower()
        self.limiter = limiter.lower() 
        self.rng = rng
    
    @partial(jax.jit, static_argnums=(0,))
    def flux(self, up, params):
        """
        up: signal_length + 2 x VariableSize
        """
        flux = jax.jit(self.Num_flux.apply)
        #d_flux = jax.jit(lambda up, params: (self.flux.apply(params, up + eps) - self.flux.apply(params, up - eps))/(2*eps))
        d_flux = jax.jit(jax.grad(lambda x, params: jnp.sum(self.Num_flux.apply(params,x))))
        return flux({'params': params['flux']},up), d_flux(up, {'params': params['flux']})        
        
    @partial(jax.jit, static_argnums=(0,))
    def Kurganov_Tadmor(self, u, params):
        
        uL, uR =jax.vmap(jax.vmap(self.linearExtrapolation, in_axes=(1), out_axes=(1)), in_axes=(0))(u)
        """
        uL, uR: signal_length + 2 x VariableSize
        """
        fL, dfL = self.flux(uL, params)
        fR, dfR = self.flux(uR, params)
        rho = jnp.maximum(jnp.abs(dfL), jnp.abs(dfR))
        H = 0.5*(fR + fL - rho*(uR - uL))
        Diff = jax.vmap(lambda H:-(H[1:] - H[:-1])/self.dx)
        y = Diff(H)
        return y

    @partial(jax.jit, static_argnums=(0,))
    def linearExtrapolation(self, u):
        """
        usize signal_length + 4 x 1
        """
        um = u[:-2]
        u_ = u[1:-1]
        up = u[2:]
        
        zeros = jnp.zeros_like(u_)
        minmod = lambda a, b: jnp.sign(b)*jnp.maximum(zeros, jnp.minimum(jnp.abs(b), jnp.sign(b)*a))
        uL = u_  + 0.5 * minmod(u_ - um, up - u_) 
        uR = u_  - 0.5 * minmod(u_ - um, up - u_) 
        return uL[:-1], uR[1:]
    
    def rhs(self, u, params):

        # self.boundary == "same":
        up = jnp.pad(u, ((0,0),(2,2),(0,0)),mode="wrap")

        return self.Kurganov_Tadmor(up, params)
              
    @partial(jax.jit, static_argnums=(0,))    
    def TVD_RK3(self, params, u):
        """
        Integrator of Runge Kutta 3 TVD
        """
        
        u1 = u + self.dt* self.rhs(u, params)
        #u1 = jnp.concatenate((u1[:,:1,:], u1[:,1:-1,:], u1[:,:1,:]), axis=1)
        u2 = 3/4*u + 1/4*u1 + 1/4*self.dt* self.rhs(u1, params)
        #u2 = jnp.concatenate((u2[:,:1,:], u2[:,1:-1,:], u2[:,:1,:]), axis=1)
        u3 = 1/3*u + 2/3*u2 + 2/3*self.dt* self.rhs(u2, params)
        #u3 = jnp.concatenate((u3[:,:1,:], u3[:,1:-1,:], u3[:,:1,:]), axis=1)
        return u3
    
    @partial(jax.jit, static_argnums=(0,))    
    def euler(self, params, u):
        """
        Integrator of Runge Kutta 3 TVD
        """
        
        u1 = u + self.dt* self.rhs(u, params)
        return u1   

@jax.jit
def apply_model(state, un, u_np1):
    """Computes gradients, loss for a single batch."""
    def loss_fn(params):
        um = un
        loss = 0
        for i in range(u_np1.shape[1]):
            u = state.apply_fn(params, um)
            # tvd = jnp.mean(jnp.abs(u[:,1:,:] - u[:,:-1,:]) - jnp.abs(um[:,1:,:] - um[:,:-1,:]))
            loss += jnp.mean((u_np1[:,i,:,:] - u)**2) #+ jnp.maximum(tvd, 0)
            # jax.debug.print("Time Step: {i}, Total Loss: {L}", i = i, L = loss)
            um  = u
        return loss
    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    return grads, loss 

@jax.jit
def apply_model_val(state, un, u_np1):
    """Computes gradients, loss for a single batch."""

    def loss_fn(params):
        um = un
        loss = 0
        for i in range(u_np1.shape[1]):
            u = state.apply_fn(params, um)
            tvd = jnp.mean(jnp.abs(u[:,1:,:] - u[:,:-1,:]) - jnp.abs(um[:,1:,:] - um[:,:-1,:]))
            loss += jnp.mean((u_np1[:,i,:,:] - u)**2) #+ jnp.maximum(tvd, 0)
            um  = u
        return loss

    return loss_fn(state.params)

@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


def train_epoch(state, train_ds, val_ds, batch_size, rng):
    """Train for a single epoch."""
    train_ds_size = len(train_ds['un'])
    steps_per_epoch = train_ds_size // batch_size

    perms = jax.random.permutation(rng, len(train_ds['un']))
    perms = perms[: steps_per_epoch * batch_size]  # skip incomplete batch
    perms = perms.reshape((steps_per_epoch, batch_size))
    
    epoch_loss = []
    count = 0
    for perm in perms:
        batch_input = train_ds['un'][perm, ...]
        batch_output = train_ds['un_p1'][perm,:,:]
        grads, loss = apply_model(state, batch_input, batch_output)
        # jax.debug.print("No. Steps {step}, Loss {l}", step = count, l=loss)
        count += 1
        state = update_model(state, grads)
        epoch_loss.append(loss)
    train_loss = np.mean(epoch_loss)
    # Validate
    val_loss = apply_model_val(state, val_ds['un'], val_ds['un_p1'])
    return state, train_loss, val_loss


def get_Datasets(Noise_level,rng=jax.random.PRNGKey(100), L = 10, data_path = 'Data/trainData.npy'):
    """Load Dataset"""
    trainData = np.load(data_path)
    train_ds = {}    
    un = []
    un_p1 = []
    for _ in range(1):
        for i in range(trainData.shape[0]):
            rng, sample_rng = jax.random.split(rng)
            index = 0 # jax.random.choice(sample_rng, 50-L)
            un.append(trainData[i:i+1,index,:,:])
            un_p1.append(trainData[i:i+1, index+1:index+L+1,:,:]) 
    un = np.concatenate(un, axis=0)
    un_p1 = np.concatenate(un_p1, axis=0)
    rng1, rng = jax.random.split(rng)
    shape = un_p1.shape
    un_p1 += jax.random.normal(rng1, shape=shape) * np.mean(np.abs(trainData)) * Noise_level
    train_ds['un'] = un # un.reshape(-1, un.shape[2],1)
    train_ds['un_p1'] = un_p1 # un_p1.reshape(-1, un_p1.shape[2], 1)
    return train_ds  



def create_train_state(ess, rng, learning_rate):
    """Creates initial `TrainState`."""
    rng1, rng2 = jax.random.split(rng)
    params1 = ess.Num_flux.init(rng1, jnp.ones([3, 1]))['params']
    params = {'flux': params1}
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=ess.TVD_RK3, params=params, tx=tx)


def TrainEntropyStableScheme(epochs, lr = 1e-4,ckpt_dir='./ckpts/Edge/'):
    
    Nx =  512
    dx = 2*np.pi/Nx
    dt = 0.005
    batch_size = 10 
    restart = True
    threshold = 1e-4
    rng = jax.random.PRNGKey(0)
    vector_rng, rng = jax.random.split(rng)
    EntropyStableForm = KurganovTadmorScheme(rng = vector_rng, Features=[64, 64, 64, 64, 64, 1], dt=dt, dx=dx, boundary="same", limiter="minmod")
    
    rng, gendata_rng = jax.random.split(rng)
    timeSteps = 20 
    Noise_level = 1.
    train_ds = get_Datasets(Noise_level, L=timeSteps, rng = gendata_rng, data_path = 'Data/trainData_Burgers_'+str(Nx)+'.npy')
    rng, gendata_rng = jax.random.split(rng)
    val_ds = get_Datasets(Noise_level, L=timeSteps, rng = gendata_rng, data_path = 'Data/valData_Burgers_' + str(Nx) + '.npy')

    #schedule = optax.piecewise_constant_schedule(lr, boundaries_and_scales={4000:0.3, 10000:0.1})
                                                 #{2000:1/2, 4000:1/2, 8000:1/2, 10000:1/2,15000:1/2})  
    schedule = optax.exponential_decay(lr, 2000, 0.95, end_value=1e-6)
    # cosine_onecycle_schedule(6000, 1e-1, pct_start=0.4, div_factor=10.0, final_div_factor=30.0)

    rng, init_rng = jax.random.split(rng)
    state = create_train_state(EntropyStableForm, init_rng, schedule)
    #orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    
    options = ocp.CheckpointManagerOptions(
        save_interval_steps=1,
        create=True
    )
    
    path = epath.Path(ckpt_dir + 'orbax/single_save')
    ckpt_dir_absolute = os.path.abspath(path)
    path = epath.Path(ckpt_dir_absolute)
    mngr = ocp.CheckpointManager(
        path, 
        options=options
    )
    if path.exists() and not restart:            
        target = state
        state_restored = mngr.restore(mngr.latest_step(), args=ocp.args.StandardRestore(target))
        state = state_restored
    else:
        shutil.rmtree(path)
        path.mkdir()
        mngr = ocp.CheckpointManager(
            path, 
            options=options
        )   

    best_val_loss = float('inf')  # Initialize the best validation loss to infinity

    for epoch in range(1, epochs + 1):
        
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
            # best_val_loss = val_loss
            # print(f'New best validation loss: {val_loss:.10f}, saving model...')
            ckpt = {'model': state}
            mngr.save(epoch, args=ocp.args.StandardSave(state.params))
            mngr.wait_until_finished()
#         if epoch % 1 == 0:
#             evaluateESS(epoch, ckpt_dir=ckpt_dir)
    
def state_shape(state):
    return jax.eval_shape(lambda: state)

def evaluateESS(epochs, ckpt_dir='ckpts/Edge/'):
    
    Nx = 512 
    dx = 2*np.pi/Nx
    dt = 0.005
    lr = 0.0001
    batch_size = 32


    rng = jax.random.PRNGKey(100)
    vector_rng, rng = jax.random.split(rng)    
    EntropyStableForm = KurganovTadmorScheme(rng = vector_rng, Features=[64, 64, 64, 64, 64, 1], dt=dt, dx=dx, boundary="same", limiter="minmod")
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(EntropyStableForm, init_rng, lr)    
    options = ocp.CheckpointManagerOptions(
    max_to_keep=3
    )
    
    path = epath.Path(ckpt_dir + 'orbax/single_save')
    ckpt_dir_absolute = os.path.abspath(path)
    path = epath.Path(ckpt_dir_absolute)
    mngr = ocp.CheckpointManager(
        path, 
        options=options
    )
    
    target = state.params 
    try:
        state_restored = mngr.restore(mngr.latest_step(), args=ocp.args.StandardRestore(target))
        state = train_state.TrainState.create(apply_fn=EntropyStableForm.TVD_RK3, params=state_restored, tx=state.tx)
        print("Checkpoint restored successfully.")
    except ValueError as e:
        print(f"Restore failed: {e}")
        return
    
    dir_path = ckpt_dir.split("/")
    if not os.path.exists('_plots/' + dir_path[1]):
        os.mkdir('_plots/' + dir_path[1])
    if not os.path.exists('_plots/' + dir_path[1] + '/Entropy'):
        os.mkdir('_plots/' + dir_path[1] + '/Entropy')   
    if not os.path.exists('_plots/' + dir_path[1] + '/u'):
        os.mkdir('_plots/' + dir_path[1] + '/u')   
    if not os.path.exists('_plots/' + dir_path[1] + '/Conserved_u'):
        os.mkdir('_plots/' + dir_path[1] + '/Conserved_u') 
    
    data_path =  'Data/testData_Burgers_'+str(Nx) +'_Low.npy'
    testData = np.load(data_path)

    
    N = 600 
    x = np.linspace(-np.pi,np.pi,Nx)
    t = np.linspace(0,N*dt,N + 1)
    
    
    un = testData[:1,0,:,:]
    un_p1 = testData[:1,1:,:,:]
    
    u = []
    Flux = []
    exact_flux = []
    u.append(un)
    Flux.append(jax.vmap(jax.vmap(EntropyStableForm.Num_flux.apply, in_axes=(None,0)), in_axes=(None,0))({'params': state.params['flux']}, un))
    for i in range(N):
        un = state.apply_fn(state.params, un)
        u.append(un)
        Flux.append(jax.vmap(jax.vmap(EntropyStableForm.Num_flux.apply, in_axes=(None,0)), in_axes=(None,0))({'params': state.params['flux']}, un))    
    un_p1 = testData[:1,0:,:,:]

    import matplotlib.pyplot as plt
    
    for j in range(N, N+1):
        pred = u[j][0,:,0]
        exact = un_p1[0,j,:,0]
        plt.figure(figsize=(4,3))
        plt.plot(x, exact, label='exact')
        plt.plot(x, pred, '-.',label='pred',lw=0.8)
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("u(x)")
        plt.title("t = " + str(j*dt) + "s")
        plt.tight_layout()
        plt.savefig("_plots/" + dir_path[1] + "/u/" + str(j).zfill(3) + ".png",dpi=200)
        plt.close()
        
        
    plt.figure(figsize=(4,3))
    plt.plot(t, [np.mean(testData[0,s,:,0]**2)/2 for s in range(testData.shape[1])], label = "Exact")
    plt.plot(t, [np.mean(s[0,:,0]**2)/2 for s in u], '-.', label = 'Pred')
    plt.xlabel('t / s')
    plt.ylabel('Entropy')
    plt.legend()
    plt.tight_layout()
    plt.savefig("_plots/" + dir_path[1] + "/Entropy/" + str(epochs).zfill(3) + ".png",dpi=200)
    plt.close()
    
    plt.figure(figsize=(4,3))
    plt.plot(t, [dx*np.sum(testData[0,s,:,0] - testData[0,0,:,0]) for s in range(testData.shape[1])], label = "Exact")
    plt.plot(t, [dx*np.sum(u[s][0,:,0] - u[0][0,:,0])- dt*(Flux[s][0,0,0] - Flux[s][0,-1,0])*s for s in range(len(u))], '-.', label = 'Pred')
    plt.xlabel('t / s')
    plt.ylabel('Conserved_U')
    plt.legend()
    plt.tight_layout()
    plt.savefig("_plots/" + dir_path[1] + "/Conserved_u/" + str(epochs).zfill(3) + ".png",dpi=200)
    plt.close()
    
if __name__=="__main__":
    #path = os.path.abspath(".")
    #print(path)
    
    TrainEntropyStableScheme(500, ckpt_dir= 'ckpts/KT_DNN_100Noise_train512_test512/')#KT_superbee_trainableSPNorm_silu_NoNoise_L15_512/')
    evaluateESS(500, ckpt_dir= 'ckpts/KT_DNN_100Noise_train512_test512/')
    
