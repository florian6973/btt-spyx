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