import spyx
import spyx.nn as snn

import jax
import jax.numpy as jnp

import haiku as hk

from collections import namedtuple

# Original code from https://github.com/IGITUGraz/eligibility_propagation
# Copyright 2019-2020, the e-prop team:
# Guillaume Bellec, Franz Scherr, Anand Subramoney, Elias Hajek, Darjan Salaj, Robert Legenstein, Wolfgang Maass
# from the Institute for theoretical computer science, TU Graz, Austria.

CustomALIFStateTuple = namedtuple('CustomALIFStateTuple', ('s', 'z', 'r', 'z_local'))

@jax.custom_gradient
def SpikeFunction(v_scaled, dampening_factor):
    z_ = jnp.greater(v_scaled, 0.)
    z_ = z_.astype(jnp.float32)

    def grad(dy):
        dE_dz = dy
        dz_dv_scaled = jnp.maximum(1 - jnp.abs(v_scaled), 0)
        dz_dv_scaled *= dampening_factor

        dE_dv_scaled = dE_dz * dz_dv_scaled

        return (dE_dv_scaled, jnp.zeros_like(dampening_factor).astype(jnp.float32))

    return z_, grad

# LIF without input and output layer
class RecurrentLIFLight(hk.RNNCore):
    """
    Leaky Integrate and Fire neuron model inspired by the implementation in
    snnTorch:

    https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html
    
    """

    def __init__(self, 
                 n_rec, tau=20., thr=.615, dt=1., dtype=jnp.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=.16, tag='',
                 stop_gradients=False, w_rec_init=None, n_refractory=1, rec=True,
                 name="RecurrentLIFLight"):
        super().__init__(name=name)

        self.n_refractory = n_refractory
        self.tau_adaptation = tau_adaptation
        self.beta = beta
        self.decay_b = jnp.exp(-dt / tau_adaptation)

        if jnp.isscalar(tau): tau = jnp.ones(n_rec, dtype=dtype) * jnp.mean(tau)
        if jnp.isscalar(thr): thr = jnp.ones(n_rec, dtype=dtype) * jnp.mean(thr)

        tau = jnp.array(tau, dtype=dtype)
        dt = jnp.array(dt, dtype=dtype)
        self.rec = rec

        self.dampening_factor = dampening_factor
        self.stop_gradients = stop_gradients
        self.dt = dt
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = jnp.exp(-dt / tau)
        self.thr = thr

        if rec:
            init_w_rec_var = w_rec_init if w_rec_init is not None else hk.initializers.TruncatedNormal(1./jnp.sqrt(n_rec))
            self.w_rec_var = hk.get_parameter("w_rec" + tag, (n_rec, n_rec), dtype, init_w_rec_var)

            self.recurrent_disconnect_mask = jnp.diag(jnp.ones(n_rec, dtype=bool))

            # Disconnect autotapse
            self.w_rec_val = jnp.where(self.recurrent_disconnect_mask, jnp.zeros_like(self.w_rec_var), self.w_rec_var)

        self.built = True

    def initial_state(self, batch_size, dtype=jnp.float32, n_rec=None):
        if n_rec is None: n_rec = self.n_rec

        s0 = jnp.zeros(shape=(batch_size, n_rec, 2), dtype=dtype)
        z0 = jnp.zeros(shape=(batch_size, n_rec), dtype=dtype)
        z_local0 = jnp.zeros(shape=(batch_size, n_rec), dtype=dtype)
        r0 = jnp.zeros(shape=(batch_size, n_rec), dtype=dtype)
        return CustomALIFStateTuple(s=s0, z=z0, r=r0, z_local=z_local0)
    
    def compute_z(self, v, b):
        adaptive_thr = self.thr + b * self.beta
        v_scaled = (v - adaptive_thr) / self.thr
        z = SpikeFunction(v_scaled, self.dampening_factor)
        z = z * 1 / self.dt
        return z
        
    def __call__(self, inputs, state, scope=None, dtype=jnp.float32):
        decay = self._decay

        z = state.z
        z_local = state.z_local
        s = state.s

        if self.stop_gradients:
            z = jax.lax.stop_gradient(z)
            # stop gradient z_local?
            
        print("iin", inputs.shape)
        i_in = inputs.reshape(-1, self.n_rec)

        if self.rec:
            if len(self.w_rec_val.shape) == 3:
                i_rec = jnp.einsum('bi,bij->bj', z, self.w_rec_val)
            else:
                i_rec = jnp.matmul(z, self.w_rec_val)

            i_t = i_in + i_rec
        else:
            i_t = i_in


        def get_new_v_b(s, i_t):
            v, b = s[..., 0], s[..., 1]
            new_b = self.decay_b * b + z_local

            I_reset = z * self.thr * self.dt
            new_v = decay * v + i_t  - I_reset

            return new_v, new_b
        
        new_v, new_b = get_new_v_b(s, i_t)

        is_refractory = state.r > 0
        zeros_like_spikes = jnp.zeros_like(z)
        new_z = jnp.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_z_local = jnp.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        new_r = state.r + self.n_refractory * new_z - 1
        new_r = jnp.clip(new_r, 0., float(self.n_refractory))

        if self.stop_gradients:
            new_r = jax.lax.stop_gradient(new_r)
        new_s = jnp.stack((new_v, new_b), axis=-1)

        new_state = CustomALIFStateTuple(s=new_s, z=new_z, r=new_r, z_local=new_z_local)
        return new_z, new_state

# Can be used for output layer   
class LeakyLinear(hk.RNNCore):
    def __init__(self, n_in, n_out, kappa, dtype=jnp.float32, name="LeakyLinear"):
        super().__init__(name=name)
        self.n_in = n_in
        self.n_out = n_out
        self.kappa = kappa

        self.dtype = dtype

        self.weights = hk.get_parameter("weights", shape=[n_in, n_out], dtype=dtype,
                                        init=hk.initializers.TruncatedNormal(1./jnp.sqrt(n_in)))

        self._num_units = self.n_out
        self.built = True


    def initial_state(self, batch_size, dtype=jnp.float32):
        s0 = jnp.zeros(shape=(batch_size, self.n_out), dtype=dtype)
        return s0

    def __call__(self, inputs, state, scope=None, dtype=jnp.float32):
        if len(self.weights.shape) == 3:
            outputs = jnp.einsum('bi,bij->bj', inputs, self.weights)
        else:
            outputs = jnp.matmul(inputs, self.weights)
        new_s = self.kappa * state  + (1 - self.kappa) * outputs
        return new_s, new_s
    

# test the network (without output layer) with MSE loss
def eval_lif_light(lsnn, inputs, w_rec, w_in, w_out, key, n_rec, dt, tau_v, T, batch_size=1):    
    from utils import exp_convolve

    lsnn_hk = hk.without_apply_rng(hk.transform(lsnn))
    i0 = jnp.stack([inputs[:,0], inputs[:,0]], axis=0)
    params = lsnn_hk.init(rng=key, x=i0, batch_size=2)
    state = None
    spikes = []
    V = []
    variations = []
    if w_rec is not None:
        params['RecurrentLIFLight']['w_rec'] = w_rec
    if w_in is not None:
        params['linear']['w'] = w_in
    if w_out is None:
        w_out = jax.random.normal(key=key, shape=[n_rec, 1]) # one output neuron
    for t in range(T):
        it = inputs[:, t]
        it = jnp.expand_dims(it, axis=0)
        outs, state = lsnn_hk.apply(params, it, state, batch_size)
        print(inputs[:,t], "->", outs)
        spikes.append(outs)
        V.append(state[0].s[...,0])
        variations.append(state[0].s[...,1])

    spikes = jnp.stack([s[0] for s in spikes], axis=0)
    spikes = jnp.expand_dims(spikes, axis=0)
    V = jnp.stack(V, axis=1)
    variations = jnp.stack(variations, axis=1)
    print(spikes.shape)
    decay_out = jnp.exp(-dt / tau_v)
    print(decay_out)
    print(spikes.shape)
    z_filtered = exp_convolve(spikes, decay_out)
    print("zf", z_filtered)
    y_out = jnp.einsum("btj,jk->tk", z_filtered, w_out) # no batch dim
    y_target = jax.random.normal(key=key, shape=[T, 1])
    print(y_out.shape, y_target.shape)
    loss = 0.5 * jnp.sum((y_out - y_target) ** 2)
    y_out = jnp.expand_dims(y_out, axis=0)
    y_target = jnp.expand_dims(y_target, axis=0)
    return loss, y_out, y_target, w_out, spikes, V, variations


# test the network with MSE loss
def eval_full(lsnn, inputs, w_rec, w_in, w_out, key, n_rec, dt, tau_v, T, batch_size=1):    
    lsnn_hk = hk.without_apply_rng(hk.transform(lsnn))
    i0 = jnp.stack([inputs[:,0], inputs[:,0]], axis=0)
    params = lsnn_hk.init(rng=key, x=i0, batch_size=2)
    state = None
    spikes = []
    V = []
    variations = []
    if w_rec is not None:
        params['RecurrentLIFLight']['w_rec'] = w_rec
    if w_in is not None:
        params['linear']['w'] = w_in
    if w_out is not None:
        params['LeakyLinear']['weights'] = w_out
    for t in range(T):
        it = inputs[:, t]
        it = jnp.expand_dims(it, axis=0)
        outs, state = lsnn_hk.apply(params, it, state, batch_size)
        print(inputs[:,t], "->", outs)
        spikes.append(outs)

    y_out = jnp.stack([s[0] for s in spikes], axis=0)
    y_target = jax.random.normal(key=key, shape=[T, 1])
    print(y_out.shape, y_target.shape)
    loss = 0.5 * jnp.sum((y_out - y_target) ** 2)
    y_out = jnp.expand_dims(y_out, axis=0)
    y_target = jnp.expand_dims(y_target, axis=0)
    return loss, y_out, y_target
    
