import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


class GaussianNoise(snt.Module):
    """ Sonnet module for adding Gaussian noise to each output. """

    def __init__(self, stddev: float, name: str = 'gaussian_noise'):
        super().__init__(name=name)
        self._noise = tfp.distributions.Normal(loc=0., scale=stddev)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        output = inputs + self._noise.sample(inputs.shape)
        return output


class EpsilonGreedyExploration(snt.Module):
    """ Sonnet module for epsilon-greedy exploration. """

    def __init__(self,
                 epsilon_init: int,
                 epsilon_final: int,
                 epsilon_schedule_timesteps: int):
        """ Initialise EpsilonGreedyExploration class.

            Parameters:
            ----------
            * epsilon_init: int
                Initial epsilon value.

            * epsilon_final: int
                Final epsilon value.
            * epsilon_schedule_timesteps: int

                Number of timesteps to decay epsilon from 'epsilon_init'
                to 'epsilon_final'

        """
        super().__init__(name='epsilon_greedy_exploration')

        # Internalise parameters.
        self.epsilon_init = epsilon_init
        self.epsilon_final = epsilon_final
        self.epsilon_schedule_timesteps = epsilon_schedule_timesteps

        # Internal counter.
        self.counter = tf.Variable(0.0)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        num_actions = tf.cast(tf.shape(inputs)[-1], inputs.dtype)

        # Dithering action distribution.
        dither_probs = 1 / num_actions * tf.ones_like(inputs)

        # Greedy action distribution, breaking ties uniformly at random.
        max_value = tf.reduce_max(inputs, axis=-1, keepdims=True)
        greedy_probs = tf.cast(tf.equal(inputs, max_value),
                            inputs.dtype)
        greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)

        # Calculate new epsilon.
        self.counter.assign(self.counter + 1.0)
        fraction = tf.math.minimum(self.counter / self.epsilon_schedule_timesteps, 1.0)
        epsilon = self.epsilon_init + fraction * (self.epsilon_final - self.epsilon_init)

        # Epsilon-greedy action distribution.
        probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

        # Construct the policy.
        policy = tfp.distributions.Categorical(probs=probs)

        # Sample from policy.
        sample = policy.sample()

        return sample
