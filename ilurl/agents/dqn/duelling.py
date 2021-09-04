import sonnet as snt
import tensorflow as tf

from typing import Sequence


class DuellingMLP(snt.Module):
  """A Duelling MLP Q-network."""

  def __init__(
      self,
      num_actions: int,
      hidden_sizes: Sequence[int],
  ):
    super().__init__(name='duelling_q_network')
    tf.config.run_functions_eagerly(True)
    self._value_mlp = snt.nets.MLP([*hidden_sizes, 1])
    self._advantage_mlp = snt.nets.MLP([*hidden_sizes, num_actions])

  def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
    """Forward pass of the duelling network.

    Args:
      inputs: 2-D tensor of shape [batch_size, embedding_size].

    Returns:
      q_values: 2-D tensor of action values of shape [batch_size, num_actions]
    """

    # Compute value & advantage for duelling.
    value = self._value_mlp(inputs)  # [B, 1]
    advantages = self._advantage_mlp(inputs)  # [B, A]

    # Advantages have zero mean.
    advantages -= tf.reduce_mean(advantages, axis=-1, keepdims=True)  # [B, A]

    q_values = value + advantages  # [B, A]
    self.q_values = q_values.numpy()
    return q_values
