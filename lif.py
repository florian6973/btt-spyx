import spyx
import spyx.nn as snn

import jax
import jax.numpy as jnp

import haiku as hk

from utils import exp_convolve, shift_by_one_time_step, iterate

from collections import namedtuple

# https://colab.research.google.com/github/google/jax/blob/main/docs/jax-101/04-advanced-autodiff.ipynb

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

class RecurrentLIF(hk.RNNCore):
    """
    Leaky Integrate and Fire neuron model inspired by the implementation in
    snnTorch:

    https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html
    
    """

    def __init__(self, 
                 n_in, n_rec, tau=20., thr=.615, dt=1., dtype=jnp.float32, dampening_factor=0.3,
                 tau_adaptation=200., beta=.16, tag='',
                 stop_gradients=False, w_in_init=None, w_rec_init=None, n_refractory=1, rec=True,
                 name="RecurrentLIF"):
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
        self.n_in = n_in
        self.n_rec = n_rec
        self.data_type = dtype

        self._num_units = self.n_rec

        self.tau = tau
        self._decay = jnp.exp(-dt / tau)
        self.thr = thr

        # init_w_in_var = w_in_init if w_in_init is not None else \
        #         (jax.random.uniform(key, shape=(n_in, n_rec)) / jnp.sqrt(n_in)).astype(dtype)
        init_w_in_var = w_in_init if w_in_init is not None else hk.initializers.TruncatedNormal(1./jnp.sqrt(n_in))
        self.w_in_var = hk.get_parameter("w_in" + tag, (n_in, n_rec), dtype, init_w_in_var)
        self.w_in_val = self.w_in_var

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

        self.variable_list = [self.w_in_var, self.w_rec_var] if rec else [self.w_in_var,]
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

        if len(self.w_in_val.shape) == 3:
            i_in = jnp.einsum('bi,bij->bj', inputs, self.w_in_val)
        else:
            # print(inputs.shape, self.w_in_val.shape)
            # i_in = jnp.matmul(inputs, self.w_in_val)
            i_in = jnp.einsum('abc,cd->ad', inputs, self.w_in_val)
            # print(inputs.shape, self.w_in_val.shape)
        print("i_in", i_in.shape)

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
    

def eval3(lsnn2, inputs, params, weights, w_in, w_out, key, n_rec, dt, tau_v, T, batch_size=1):    
    lsnn2_hk = hk.without_apply_rng(hk.transform(lsnn2))
    params = lsnn2_hk.init(rng=key, x=inputs, batch_size=1)
    state = None
    spikes = []
    V = []
    variations = []
    params['RecurrentLIF']['w_rec'] = weights
    if w_in is not None:
        params['RecurrentLIF']['w_in'] = w_in
    for t in range(T):
        it = inputs[:, t]
        it = jnp.expand_dims(it, axis=0)
        outs, state = lsnn2_hk.apply(params, it, state, batch_size)
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
    if w_out is None:
        w_out = jax.random.normal(key=key, shape=[n_rec, 1])
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
    
def compute_eligibility_traces(v_scaled, z_pre, z_post, is_rec, dt, thr, tau_a, tau_v, beta, dampening_factor):
    n_neurons = jnp.shape(z_post)[2]
    rho = jnp.exp(-dt / tau_a)
    # beta = beta # defined outside
    alpha = jnp.ones(z_post.shape[-1])*jnp.exp(-dt/tau_v)
    n_ref = 3 #n_refractory

    # everything should be time major
    # z_pre = tf.transpose(z_pre, perm=[1, 0, 2])
    # v_scaled = tf.transpose(v_scaled, perm=[1, 0, 2])
    # z_post = tf.transpose(z_post, perm=[1, 0, 2])

    z_pre = jax.lax.transpose(z_pre, permutation=[1, 0, 2])
    v_scaled = jax.lax.transpose(v_scaled, permutation=[1, 0, 2])
    z_post = jax.lax.transpose(z_post, permutation=[1, 0, 2])

    psi_no_ref = dampening_factor / thr * jnp.maximum(0., 1. - jnp.abs(v_scaled))

    update_refractory = lambda refractory_count, z_post:\
        jnp.where(z_post > 0,jnp.ones_like(refractory_count) * (n_ref - 1),jnp.maximum(0, refractory_count - 1))

    refractory_count_init = jnp.zeros_like(z_post[0], dtype=jnp.int32)
    
    # refractory_count = [refractory_count_init]
    # for z in z_post[:-1]:
    #     refractory_count.append(update_refractory(refractory_count[-1], z))
    # print(jnp.array(refractory_count).shape)
    refractory_count = iterate(update_refractory, z_post[:-1], refractory_count_init)

    print("refractory_count", refractory_count)
    # refractory_count = tf.scan(update_refractory, z_post[:-1], initializer=refractory_count_init)
    # print(refractory_count_init.shape, z_post[:-1].shape)
    # print(update_refractory(update_refractory(refractory_count_init, z_post[0]), z_post[1]))
    # print(jax.lax.scan(update_refractory, xs=z_post[:-1], init=refractory_count_init))
    # _, refractory_count = jax.lax.scan(update_refractory, xs=z_post[:-1], init=refractory_count_init)
    # refractory_count = jnp.concat([jnp.expand_dims(refractory_count_init, axis=0), refractory_count], axis=0)

    is_refractory = refractory_count > 0
    psi = jnp.where(is_refractory, jnp.zeros_like(psi_no_ref), psi_no_ref)

    print("psi", psi)

    update_epsilon_v = lambda epsilon_v, z_pre: alpha[None, None, :] * epsilon_v + z_pre[:, :, None] #alpha[None, None, :] * epsilon_v + z_pre[:, :, None]
    epsilon_v_zero = jnp.ones((1, 1, n_neurons)) * z_pre[0][:, :, None]
    print("evz", epsilon_v_zero)
    print("zpre", z_pre[1:])
    # epsilon_v = tf.scan(update_epsilon_v, z_pre[1:], initializer=epsilon_v_zero, )
    # _, epsilon_v = jax.lax.scan(update_epsilon_v, xs=z_pre[1:], init=epsilon_v_zero)
    # epsilon_v = jnp.concat([[epsilon_v_zero], epsilon_v], axis=0)
    print(epsilon_v_zero.shape, z_pre[1:].shape)
    epsilon_v = iterate(update_epsilon_v, z_pre[1:], epsilon_v_zero)
    print("ev", epsilon_v)

    update_epsilon_a = lambda epsilon_a, elems:\
            (rho - beta * elems['psi'][:, None, :]) * epsilon_a + elems['psi'][:, None, :] * elems['epsi']

    epsilon_a_zero = jnp.zeros_like(epsilon_v[0])
    # epsilon_a = tf.scan(fn=update_epsilon_a,
    #                     elems={'psi': psi[:-1], 'epsi': epsilon_v[:-1], 'previous_epsi':shift_by_one_time_step(epsilon_v[:-1])},
    #                     initializer=epsilon_a_zero)
    # _, epsilon_a = jax.lax.scan(update_epsilon_a,
    #                             xs=[{'psi': psi[:-1], 'epsi': epsilon_v[:-1], 'previous_epsi':shift_by_one_time_step(epsilon_v[:-1])}],
    #                             init=epsilon_a_zero)

    # epsilon_a = jnp.concat([[epsilon_a_zero], epsilon_a], axis=0)
    # epsilon_a = iterate(update_epsilon_a, [{'psi': psi[:-1], 'epsi': epsilon_v[:-1], 'previous_epsi':shift_by_one_time_step(epsilon_v[:-1])}], epsilon_a_zero)
    previous_epsi = shift_by_one_time_step(epsilon_v[:-1])
    elems = [{'psi': psie, 'epsi': epsie, 'previous_epsi': pe} for psie, epsie, pe in zip(psi[:-1], epsilon_v[:-1], previous_epsi)]
    epsilon_a = iterate(update_epsilon_a, elems, epsilon_a_zero)
    print("ea", epsilon_a)

    e_trace = psi[:, :, None, :] * (epsilon_v - beta * epsilon_a)

    # everything should be time major
    # e_trace = tf.transpose(e_trace, perm=[1, 0, 2, 3])
    # epsilon_v = tf.transpose(epsilon_v, perm=[1, 0, 2, 3])
    # epsilon_a = tf.transpose(epsilon_a, perm=[1, 0, 2, 3])
    # psi = tf.transpose(psi, perm=[1, 0, 2])

    e_trace = jax.lax.transpose(e_trace, permutation=[1, 0, 2, 3])
    epsilon_v = jax.lax.transpose(epsilon_v, permutation=[1, 0, 2, 3])
    epsilon_a = jax.lax.transpose(epsilon_a, permutation=[1, 0, 2, 3])
    psi = jax.lax.transpose(psi, permutation=[1, 0, 2])

    if is_rec:
        identity_diag = jnp.eye(n_neurons)[None, None, :, :]
        e_trace -= identity_diag * e_trace
        epsilon_v -= identity_diag * epsilon_v
        epsilon_a -= identity_diag * epsilon_a

    return e_trace, epsilon_v, epsilon_a, psi

def compute_loss_gradient(learning_signal, z_pre, z_post, v_post, b_post,
                            dt, thr, tau_a, tau_v, beta, dampening_factor,
                              decay_out=None,zero_on_diagonal=None):
        thr_post = thr + beta * b_post
        v_scaled = (v_post - thr_post) / thr
        print(v_scaled)

        e_trace, epsilon_v, epsilon_a, _ = compute_eligibility_traces(v_scaled, z_pre, z_post, zero_on_diagonal,
                                                                      dt, thr, tau_a, tau_v, beta, dampening_factor)
        print("evbv", epsilon_v)
        print("eabv", epsilon_a)

        if decay_out is not None:
            e_trace_time_major = jax.lax.transpose(e_trace, permutation=[1, 0, 2, 3])
            filtered_e_zero = jnp.zeros_like(e_trace_time_major[0])
            filtering = lambda filtered_e, e: filtered_e * decay_out + e * (1 - decay_out)
            # filtered_e = tf.scan(filtering, e_trace_time_major, initializer=filtered_e_zero)
            # _, filtered_e = jax.lax.scan(filtering, xs=e_trace_time_major, init=filtered_e_zero)
            filtered_e = iterate(filtering, e_trace_time_major, filtered_e_zero, remove_first=True)
            filtered_e = jax.lax.transpose(filtered_e, permutation=[1, 0, 2, 3])
            e_trace = filtered_e
        print("e_trace", e_trace)

        if len(learning_signal.shape) == 2:
            learning_signal = jnp.expand_dims(learning_signal, axis=0)

        print(e_trace.shape, learning_signal.shape)

        gradient = jnp.einsum('btj,btij->ij', learning_signal, e_trace)
        return gradient, e_trace, epsilon_v, epsilon_a