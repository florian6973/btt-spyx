# Spiking Neural Network - e-prop

Reimplementing the e-prop framework with jax and spyx.

Report: [here](./Report.pdf)

## Dependencies

- Python 3.12
- spyx
- tonic
- nir
- torchvision

## File organization

- `lif_light.py`: spyx implementation of neuron models
- `lif.py`: basic adaptation of the original tensorflow code
- `autodiff.ipynb`: verification of e-prop calculation
- `shd.ipynb`: demo of e-prop on the SHD dataset
- `neuron_types.ipynb`: playing around with neuron models
- `test_autodiff.ipynb`: a few tests
- `utils.py`: some useful methods

> Original paper code repo: https://github.com/IGITUGraz/eligibility_propagation

> Full paper: A solution to the learning dilemma for recurrent networks of spiking neurons.
> G Bellec*, F Scherr*, A Subramoney, E Hajek, Darjan Salaj, R Legenstein, W Maass

I copied part of the original code in the `original_code` folder, to check the results with jax. To run it, install tensorflow 2 (I adapted the code for compatibility).

Other implementations of e-prop can be find at https://github.com/YigitDemirag/eprop-jax (out of spyx framework, no autodiff) or https://github.com/ChFrenkel/eprop-PyTorch (pytorch).
