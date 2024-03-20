import spyx
import spyx.nn as snn

import jax
import jax.numpy as jnp

import haiku as hk

from utils import exp_convolve, shift_by_one_time_step, iterate
from lif import SpikeFunction, CustomALIFStateTuple

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

        # init_w_in_var = w_in_init if w_in_init is not None else \
        #         (jax.random.uniform(key, shape=(n_in, n_rec)) / jnp.sqrt(n_in)).astype(dtype)
        # init_w_in_var = w_in_init if w_in_init is not None else hk.initializers.TruncatedNormal(1./jnp.sqrt(n_in))
        # self.w_in_var = hk.get_parameter("w_in" + tag, (n_in, n_rec), dtype, init_w_in_var)
        # self.w_in_val = self.w_in_var

        if rec:
            # init_w_rec_var = w_rec_init if w_rec_init is not None else \
            # (jax.random.uniform(key, shape=(n_rec, n_rec)) / jnp.sqrt(n_rec)).astype(dtype)
            init_w_rec_var = w_rec_init if w_rec_init is not None else hk.initializers.TruncatedNormal(1./jnp.sqrt(n_rec))
            self.w_rec_var = hk.get_parameter("w_rec" + tag, (n_rec, n_rec), dtype, init_w_rec_var)

            self.recurrent_disconnect_mask = jnp.diag(jnp.ones(n_rec, dtype=bool))

            # Disconnect autotapse
            self.w_rec_val = jnp.where(self.recurrent_disconnect_mask, jnp.zeros_like(self.w_rec_var), self.w_rec_var)

            # dw_val_dw_var_rec = jnp.ones((self._num_units,self._num_units)) - jnp.diag(jnp.ones(self._num_units))
        
        # dw_val_dw_var_in = jnp.ones((n_in,self._num_units))

        # self.dw_val_dw_var = [dw_val_dw_var_in, dw_val_dw_var_rec] if rec else [dw_val_dw_var_in,]

        # self.variable_list = [self.w_in_var, self.w_rec_var] if rec else [self.w_in_var,]
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
        # print("zs", z.shape, s.shape)
        # return

        if self.stop_gradients:
            z = jax.lax.stop_gradient(z)

        # if len(self.w_in_val.shape) == 3:
        #     i_in = jnp.einsum('bi,bij->bj', inputs, self.w_in_val)
        # else:
        #     # print(inputs.shape, self.w_in_val.shape)
        #     # i_in = jnp.matmul(inputs, self.w_in_val)
        #     i_in = jnp.einsum('abc,cd->ad', inputs, self.w_in_val)
        #     # print(inputs.shape, self.w_in_val.shape)
        #     # print("i_in", i_in.shape)
            
        print("iin", inputs.shape)
        i_in = inputs.reshape(-1, self.n_rec)

        if self.rec:
            if len(self.w_rec_val.shape) == 3:
                i_rec = jnp.einsum('bi,bij->bj', z, self.w_rec_val)
            else:
                # print("z wrec", z.shape, self.w_rec_val.shape)
                i_rec = jnp.matmul(z, self.w_rec_val)

            i_t = i_in + i_rec
        else:
            i_t = i_in

        # print("i_t", i_t.shape)


        def get_new_v_b(s, i_t):
            v, b = s[..., 0], s[..., 1]
            # print("vs", v.shape, b.shape)
            # old_z = self.compute_z(v, b)
            new_b = self.decay_b * b + z_local #old_z

            I_reset = z * self.thr * self.dt
            # print('vii', v.shape, i_t.shape, I_reset.shape)
            new_v = decay * v + i_t  - I_reset

            return new_v, new_b
        
        new_v, new_b = get_new_v_b(s, i_t)
        # print("nv nb", new_v.shape, new_b.shape)


        # is_refractory = jnp.greater(state.r, .1)
        # zeros_like_spikes = jnp.zeros_like(state.z)
        # new_z = jnp.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))
        # new_z_local = jnp.where(is_refractory, zeros_like_spikes, self.compute_z(new_v, new_b))

        # new_r = jnp.clip(state.r + self.n_refractory * new_z - 1,
        #                          0., float(self.n_refractory))
        # new_s = jnp.stack((new_v, new_b), axis=-1)

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
    

def eval_lif_light(lsnn, inputs, w_rec, w_in, w_out, key, n_rec, dt, tau_v, T, batch_size=1):    
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
        w_out = jax.random.normal(key=key, shape=[n_rec, 1])
    # if w_in is not None:
    #     params['RecurrentLIF']['w_in'] = w_in
    for t in range(T):
        it = inputs[:, t]
        it = jnp.expand_dims(it, axis=0)
        outs, state = lsnn_hk.apply(params, it, state, batch_size)
        # print(state[0].s.shape)
        print(inputs[:,t], "->", outs)
        spikes.append(outs)
        V.append(state[0].s[...,0])
        variations.append(state[0].s[...,1])
        # print(V[-1].shape)

    spikes = jnp.stack([s[0] for s in spikes], axis=0)
    spikes = jnp.expand_dims(spikes, axis=0)
    # spikes = jnp.stack(spikes, axis=0)
    V = jnp.stack(V, axis=1)
    variations = jnp.stack(variations, axis=1)
    print(spikes.shape)
    # w_out = jax.random.normal(key=key, shape=[n_rec, 1])
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
    
