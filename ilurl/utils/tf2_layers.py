import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp


class GaussianNoiseExploration(snt.Module):
    """ Sonnet module for adding gaussian noise (exploration). """

    def __init__(self,
                 eval_mode: bool,
                 stddev_init: float = 0.4,
                 stddev_final: float = 0.01,
                 stddev_schedule_timesteps: int = 25000,
                 ):
        """ Initialise GaussianNoise class.

            Parameters:
            ----------
            * eval_mode: bool
                If eval_mode is True then this module does not affect
                input values.

            * stddev_init: int
                Initial stddev value.

            * stddev_final: int
                Final stddev value.

            * stddev_schedule_timesteps: int
                Number of timesteps to decay stddev from 'stddev_init'
                to 'stddev_final'

        """
        super().__init__(name='gaussian_noise_exploration')

        # Internalise parameters.
        self._stddev_init = stddev_init
        self._stddev_final = stddev_final
        self._stddev_schedule_timesteps = stddev_schedule_timesteps
        self._eval_mode = tf.Variable(eval_mode)

        # Internal counter.
        self._counter = tf.Variable(0.0)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        # Calculate new stddev value.
        self._counter.assign(self._counter + 1.0)
        fraction = tf.math.minimum(self._counter / self._stddev_schedule_timesteps, 1.0)
        stddev = self._stddev_init + fraction * (self._stddev_final - self._stddev_init)

        # Add noise. If eval_mode is True then no noise is added. If
        # eval_mode is False (training mode) then gaussian noise is added to the inputs.
        noise = tf.where(self._eval_mode,
                        tf.zeros_like(inputs),
                        tfp.distributions.Normal(loc=0., scale=stddev).sample(inputs.shape))

        # Add noise to inputs.
        output = inputs + noise

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
        self._epsilon_init = epsilon_init
        self._epsilon_final = epsilon_final
        self._epsilon_schedule_timesteps = epsilon_schedule_timesteps

        # Internal counter.
        self._counter = tf.Variable(0.0)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        num_actions = tf.cast(tf.shape(inputs)[-1], inputs.dtype)

        # Dithering action distribution.
        dither_probs = 1 / num_actions * tf.ones_like(inputs)

        # Greedy action distribution, breaking ties uniformly at random.
        max_value = tf.reduce_max(inputs, axis=-1, keepdims=True)
        greedy_probs = tf.cast(tf.equal(inputs, max_value),
                            inputs.dtype)
        greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)

        # Calculate new epsilon value.
        self._counter.assign(self._counter + 1.0)
        fraction = tf.math.minimum(self._counter / self._epsilon_schedule_timesteps, 1.0)
        epsilon = self._epsilon_init + fraction * (self._epsilon_final - self._epsilon_init)

        # Epsilon-greedy action distribution.
        probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

        # Construct the policy.
        policy = tfp.distributions.Categorical(probs=probs)

        # Sample from policy.
        sample = policy.sample()

        return sample

@snt.allow_empty_variables
class InputStandardization(snt.Module):
    """ Sonnet module to scale inputs. """

    def __init__(self, shape):
        """ Initialise InputStandardization class.
            Parameters:
            ----------
            * shape: state space shape.
        """
        super().__init__(name='normalization')

        # Internalise parameters.
        self._mean = tf.Variable(tf.zeros(shape=shape), trainable=False)
        self._var = tf.Variable(tf.ones(shape=shape), trainable=False)
        self._count = tf.Variable(1e-4, trainable=False)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:

        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0])
        batch_count = tf.cast(tf.shape(inputs)[0], inputs.dtype)

        if batch_count > 1:
            # Update moving average and std.
            delta = batch_mean - self._mean
            tot_count = self._count + batch_count

            self._mean.assign(self._mean + delta * batch_count / tot_count)
            m_a = self._var * self._count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + tf.math.square(delta) * self._count * batch_count / tot_count
            self._var.assign(M2 / tot_count)
            self._count.assign(tot_count)

        # Standardize inputs.
        normalized = (inputs - self._mean) / self._var

        return normalized 