import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

""" 
    Noise layers (for exploration).
"""
class GaussianNoise(snt.Module):
  """ Sonnet module for adding Gaussian noise to each output."""

  def __init__(self, stddev: float, name: str = 'gaussian_noise'):
    super().__init__(name=name)
    self._noise = tfd.Normal(loc=0., scale=stddev)

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    output = inputs + self._noise.sample(inputs.shape)
    return output