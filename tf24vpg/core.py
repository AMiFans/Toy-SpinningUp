import numpy as np
import tensorflow as tf
from gym.spaces import Box, Discrete
import scipy.signal

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)
    
def statistics_scalar(x):
    x = np.array(x, dtype=np.float32)
    return np.mean(x), np.std(x)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
'''
def discount_cumsum(x, discount_factor, print_info=None):

    size = x.shape[0]
    if print_info is not None:
        print('Input shape', size, 'Discount_factor', discount_factor)
    discount_sum = np.zeros((size,))
    # x[::-1] is reverse of x
    for idx, value in enumerate(x[::-1]):
        discount_sum[:size - idx] += value
        if size - idx - 1 == 0:
            break
        discount_sum[:size - idx - 1] *= discount_factor
        return discount_sum
'''
def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x - mu) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

class MLP(tf.keras.layers.Layer):
    def __init__(self, sizes=(32,), activation=tf.tanh):
        super(MLP, self).__init__()
        self.denses = [tf.keras.layers.Dense(size, activation=activation) for size in sizes[:-1]]
        self.out = tf.keras.layers.Dense(sizes[-1])

    def call(self, inputs):
        x = inputs
        for dense in self.denses:
            x = dense(x)
        return self.out(x)

class CategoricalPolicy(tf.keras.layers.Layer):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(CategoricalPolicy, self).__init__()
        self.act_dim = act_dim
        self.logits = MLP(list(hidden_sizes) + [self.act_dim], activation)

    def call(self, inputs):
        obs, act = inputs
        logits = self.logits(obs)
        log_p_all = tf.nn.log_softmax(logits)
        pi = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
        if act is not None:
            act = tf.cast(act, dtype=tf.int32)
            logp_pi = tf.reduce_sum(tf.one_hot(act, depth=self.act_dim) * log_p_all, axis=1)
        else:
            logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=self.act_dim) * log_p_all, axis=1)

        return pi, logp_pi

class GaussianPolicy(tf.keras.layers.Layer):
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super(GaussianPolicy, self).__init__()
        self.mu = MLP(list(hidden_sizes) + [act_dim], activation)
        self.log_std = tf.Variable(name='log_std', initial_value=-0.5 * np.ones(act_dim, dtype=np.float32))

    def call(self, inputs):
        obs, act = inputs
        mu = self.mu(obs)
        std = tf.exp(self.log_std)
        pi = mu + tf.random.normal(tf.shape(mu)) * std  # pi is the next action
        if act is not None:
            logp_pi = gaussian_likelihood(act, mu, self.log_std)
        else:
            logp_pi = gaussian_likelihood(pi, mu, self.log_std)
        return pi, logp_pi

# Actor-Critics
class ActorCritic(tf.keras.Model):
    def __init__(self, observation_space, action_space, hidden_sizes=(64,64), activation=tf.tanh):
        super(ActorCritic, self).__init__()
        obs_dim = observation_space.shape[0]

        if isinstance(action_space, Box):
            self.policy = GaussianPolicy(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.policy = CategoricalPolicy(obs_dim, action_space.n, hidden_sizes, activation)

        self.v_mlp = MLP(list(hidden_sizes) + [1], activation)

    def call(self, inputs):
        obs, act = inputs
        pi, logp_pi = self.policy((obs, act))
        v = tf.squeeze(self.v_mlp(obs), axis=1)
        return pi, logp_pi, v
