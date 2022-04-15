from tensorflow.keras.layers import Layer
import tensorflow as tf
import numpy as np


def uniform(shape, scale=0.05, name=None):
    """Uniform init."""
    initial = tf.random_uniform_initializer(shape, minval=-scale, maxval=scale, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def glorot(shape, name=None):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform_initializer(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def zeros(shape, name=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def dot(x, y, sparse=False):
    """Wrapper for 3D tf.matmul (sparse vs dense)."""
    res = tf.einsum('bij,jk->bik', x, y)  # tf.matmul(x, y)
    return res


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform_initializer(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def gru_unit(support, x, var, act, mask, dropout, sparse_inputs=False):
    """GRU unit with 3D tensor inputs."""
    # message passing
    support = tf.nn.dropout(support, dropout)  # optional
    a = tf.matmul(support, x)

    # update gate
    z0 = dot(a, var['weights_z0'], sparse_inputs) + var['bias_z0']
    z1 = dot(x, var['weights_z1'], sparse_inputs) + var['bias_z1']
    z = tf.sigmoid(z0 + z1)

    # reset gate
    r0 = dot(a, var['weights_r0'], sparse_inputs) + var['bias_r0']
    r1 = dot(x, var['weights_r1'], sparse_inputs) + var['bias_r1']
    r = tf.sigmoid(r0 + r1)

    # update embeddings
    h0 = dot(a, var['weights_h0'], sparse_inputs) + var['bias_h0']
    h1 = dot(r * x, var['weights_h1'], sparse_inputs) + var['bias_h1']
    h = act(mask * (h0 + h1))

    return h * z + x * (1 - z)


class Dense(Layer):
    """Dense layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0., sparse_inputs=False,
                 act=tf.nn.relu, bias=False, featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights'] = glorot([input_dim, output_dim],
                                          name='weights')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # transform
        output = dot(x, self.vars['weights'], sparse=self.sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.act(output)


class GraphLayer(Layer):
    """Graph layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, steps=2, **kwargs):
        super(GraphLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.mask = placeholders['mask']
        self.steps = steps

        # helper variable for sparse dropout
        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.num_features_nonzero = placeholders['num_features_nonzero']

            self.vars['weights_encode'] = glorot([input_dim, output_dim],
                                                 name='weights_encode')
            self.vars['weights_z0'] = glorot([output_dim, output_dim], name='weights_z0')
            self.vars['weights_z1'] = glorot([output_dim, output_dim], name='weights_z1')
            self.vars['weights_r0'] = glorot([output_dim, output_dim], name='weights_r0')
            self.vars['weights_r1'] = glorot([output_dim, output_dim], name='weights_r1')
            self.vars['weights_h0'] = glorot([output_dim, output_dim], name='weights_h0')
            self.vars['weights_h1'] = glorot([output_dim, output_dim], name='weights_h1')

            self.vars['bias_encode'] = zeros([output_dim], name='bias_encode')
            self.vars['bias_z0'] = zeros([output_dim], name='bias_z0')
            self.vars['bias_z1'] = zeros([output_dim], name='bias_z1')
            self.vars['bias_r0'] = zeros([output_dim], name='bias_r0')
            self.vars['bias_r1'] = zeros([output_dim], name='bias_r1')
            self.vars['bias_h0'] = zeros([output_dim], name='bias_h0')
            self.vars['bias_h1'] = zeros([output_dim], name='bias_h1')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1 - self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1 - self.dropout)

        # encode inputs
        x = dot(x, self.vars['weights_encode'],
                self.sparse_inputs) + self.vars['bias_encode']   # (32, 50, 3)
        output = self.mask * self.act(x)

        # convolve
        for _ in range(self.steps):
            output = gru_unit(self.support, output, self.vars, self.act,
                              self.mask, 1 - self.dropout, self.sparse_inputs)

        return output


class ReadoutLayer(Layer):
    """Graph Readout Layer."""

    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        super(ReadoutLayer, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.sparse_inputs = sparse_inputs
        self.bias = bias
        self.mask = placeholders['mask']

        with tf.compat.v1.variable_scope(self.name + '_vars'):
            self.vars['weights_att'] = glorot([input_dim, 1], name='weights_att')
            self.vars['weights_emb'] = glorot([input_dim, input_dim], name='weights_emb')
            self.vars['weights_mlp'] = glorot([input_dim, output_dim], name='weights_mlp')

            self.vars['bias_att'] = zeros([1], name='bias_att')
            self.vars['bias_emb'] = zeros([input_dim], name='bias_emb')
            self.vars['bias_mlp'] = zeros([output_dim], name='bias_mlp')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs

        # soft attention
        att = tf.sigmoid(dot(x, self.vars['weights_att']) + self.vars['bias_att'])
        emb = self.act(dot(x, self.vars['weights_emb']) + self.vars['bias_emb'])

        N = tf.reduce_sum(self.mask, axis=1)
        M = (self.mask - 1) * 1e9

        # graph summation
        g = self.mask * att * emb
        g = tf.reduce_sum(g, axis=1) / N + tf.reduce_max(g + M, axis=1)
        g = tf.nn.dropout(g, 1 - self.dropout)

        # classification
        output = tf.matmul(g, self.vars['weights_mlp']) + self.vars['bias_mlp']

        return output


if __name__ == '__main__':
    pass