import spyx
import spyx.nn as snn

import jax
import jax.numpy as jnp

import haiku as hk



def exp_convolve(tensor, decay):
    '''
    Filters a tensor with an exponential filter.
    :param tensor: a tensor of shape (trial, time, neuron)
    :param decay: a decay constant of the form exp(-dt/tau) with tau the time constant
    :return: the filtered tensor of shape (trial, time, neuron)
    '''
    r_shp = range(len(tensor.shape))
    transpose_perm = [1, 0] + list(r_shp)[2:]

    tensor_time_major = jax.lax.transpose(tensor, permutation=transpose_perm)
    initializer = jnp.zeros_like(tensor_time_major[0])
    # returns a tuple scan
    # fun = lambda a, x: (a * decay + (1 - decay) * x, a * decay + (1 - decay) * x)
    # _, filtered_tensor = jax.lax.scan(fun, xs=tensor_time_major, init=initializer)

    fun = lambda a, x: a * decay + (1 - decay) * x
    print(tensor_time_major.shape, initializer.shape, decay)
    filtered_tensor = iterate(fun, tensor_time_major, initializer, remove_first=True)
    filtered_tensor = jax.lax.transpose(filtered_tensor, permutation=transpose_perm)

    return filtered_tensor

def shift_by_one_time_step(tensor, initializer=None):
    '''
    Shift the input on the time dimension by one.
    :param tensor: a tensor of shape (trial, time, neuron)
    :param initializer: pre-prend this as the new first element on the time dimension
    :return: a shifted tensor of shape (trial, time, neuron)
    '''
    if len(tensor.shape) == 2:
        tensor = jnp.expand_dims(tensor, axis=0)
    r_shp = range(len(tensor.shape))
    transpose_perm = [1, 0] + list(r_shp)[2:]
    tensor_time_major = jax.lax.transpose(tensor, permutation=transpose_perm)

    if initializer is None:
        initializer = jnp.zeros_like(tensor_time_major[0])

    # print(initializer.shape, tensor_time_major[:,:-1].shape)

    shifted_tensor = jnp.concat([initializer[None, :, :], tensor_time_major[:-1]], axis=0)
    # shifted_tensor = tensor_time_major
    # shifted_tensor = jnp.concatenate([initializer[:], tensor_time_major[:,:-1]], axis=1)

    shifted_tensor = jax.lax.transpose(shifted_tensor, permutation=transpose_perm)
    return shifted_tensor

def iterate(fun, xs, init, remove_first=False):
    rets = [init]
    for x in xs:
        # print(rets[-1].shape, x.shape)
        rets.append(fun(rets[-1], x))
    return jnp.array(rets if not remove_first else rets[1:])