import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.keras import initializers, regularizers, constraints
import numpy as np


def dot(x, y):
    """Wrapper for 3D tf.matmul (sparse vs dense)."""
    res = tf.einsum('bij,jk->bik', x, y)  # tf.matmul(x, y)
    return res


class GraphLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim

        self.init = initializers.get('glorot_uniform')
        self.zero_init = initializers.get('zeros')
        self.W_regularizer = regularizers.get()
        self.b_regularizer = regularizers.get()
        self.W_constraint = constraints.get()
        self.b_constraint = constraints.get()

        super(GraphLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        batch_size, imput_dim, features_dim = input_shape[0], input_shape[1], input_shape[2]

        self.z0 = self.add_weight(shape=(self.output_dim, self.output_dim),
                                  initializer=self.init,
                                  name='weights_z0',
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)
        self.z1 = self.add_weight(shape=(self.output_dim, self.output_dim),
                                  initializer=self.init,
                                  name='weights_z1',
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)
        self.r0 = self.add_weight(shape=(self.output_dim, self.output_dim),
                                  initializer=self.init,
                                  name='weights_r0',
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)
        self.r1 = self.add_weight(shape=(self.output_dim, self.output_dim),
                                  initializer=self.init,
                                  name='weights_r1',
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)
        self.h0 = self.add_weight(shape=(self.output_dim, self.output_dim),
                                  initializer=self.init,
                                  name='weights_h0',
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)
        self.h1 = self.add_weight(shape=(self.output_dim, self.output_dim),
                                  initializer=self.init,
                                  name='weights_h1',
                                  regularizer=self.W_regularizer,
                                  constraint=self.W_constraint)

        self.bias_z0 = self.add_weight(shape=(self.output_dim,),
                                       initializer=self.zero_init,
                                       name='bias_z0',
                                       regularizer=self.W_regularizer,
                                       constraint=self.W_constraint)
        self.bias_z1 = self.add_weight(shape=(self.output_dim,),
                                       initializer=self.zero_init,
                                       name='bias_z1',
                                       regularizer=self.W_regularizer,
                                       constraint=self.W_constraint)
        self.bias_r0 = self.add_weight(shape=(self.output_dim,),
                                       initializer=self.zero_init,
                                       name='bias_r0',
                                       regularizer=self.W_regularizer,
                                       constraint=self.W_constraint)
        self.bias_r1 = self.add_weight(shape=(self.output_dim,),
                                       initializer=self.zero_init,
                                       name='bias_r1',
                                       regularizer=self.W_regularizer,
                                       constraint=self.W_constraint)
        self.bias_h0 = self.add_weight(shape=(self.output_dim,),
                                       initializer=self.zero_init,
                                       name='bias_h0',
                                       regularizer=self.W_regularizer,
                                       constraint=self.W_constraint)
        self.bias_h1 = self.add_weight(shape=(self.output_dim,),
                                       initializer=self.zero_init,
                                       name='bias_h1',
                                       regularizer=self.W_regularizer,
                                       constraint=self.W_constraint)

        self.built = True

    def call(self, inputs):
        """
        暂定输入为[adj_mask, adj, features]
        :param inputs:
        :return:
        """
        mask, adj, x = inputs
        a = tf.matmul(adj, x)
        # adj = tf.nn.dropout(adj, 0.5)

        # forget gate
        z0 = dot(a, self.z0) + self.bias_z0
        z1 = dot(x, self.z1) + self.bias_z1
        z = K.sigmoid(z0 + z1)

        # reset gate
        r0 = dot(a, self.r0) + self.bias_r0
        r1 = dot(x, self.r1) + self.bias_r1
        r = K.sigmoid(r0 + r1)

        # update embeddings
        h0 = dot(a, self.h0) + self.bias_h0
        h1 = dot(r * x, self.h1) + self.bias_h1
        h = K.relu(mask * (h0 + h1))

        return h * z + x * (1 - z)


class ReadoutLayer(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim

        self.init = initializers.get('glorot_uniform')
        self.zero_init = initializers.get('zeros')
        self.W_regularizer = regularizers.get()
        self.b_regularizer = regularizers.get()
        self.W_constraint = constraints.get()
        self.b_constraint = constraints.get()

        super(ReadoutLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.att = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   initializer=self.init,
                                   name='weights_att',
                                   regularizer=self.W_regularizer,
                                   constraint=self.W_constraint)
        self.emb = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   initializer=self.init,
                                   name='weights_emb',
                                   regularizer=self.W_regularizer,
                                   constraint=self.W_constraint)
        self.mlp = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   initializer=self.init,
                                   name='weights_mlp',
                                   regularizer=self.W_regularizer,
                                   constraint=self.W_constraint)

        self.bias_att = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.zero_init,
                                        name='bias_att',
                                        regularizer=self.W_regularizer,
                                        constraint=self.W_constraint)
        self.bias_emb = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.zero_init,
                                        name='bias_emb',
                                        regularizer=self.W_regularizer,
                                        constraint=self.W_constraint)
        self.bias_mlp = self.add_weight(shape=(self.output_dim,),
                                        initializer=self.zero_init,
                                        name='bias_mlp',
                                        regularizer=self.W_regularizer,
                                        constraint=self.W_constraint)
        self.built = True

    def call(self, inputs):
        """
        暂定输入为[adj_mask, x]
        :param inputs:
        :return:
        """
        mask, x = inputs

        # soft attention
        att = K.sigmoid(dot(x, self.att) + self.bias_att)
        emb = K.tanh(dot(x, self.emb) + self.bias_emb)

        N = tf.reduce_sum(self.mask, axis=1)
        M = (mask - 1) * 1e9

        # graph summation
        g = mask * att * emb
        g = tf.reduce_sum(g, axis=1) / N + tf.reduce_max(g + M, axis=1)
        # g = tf.nn.dropout(g, 0.5)

        # classification
        output = tf.matmul(g, self.mlp) + self.bias_mlp

        return output


if __name__ == '__main__':
    mask = np.random.uniform(0, 1, (32, 50, 1))
    adj = np.random.uniform(0, 1, (32, 50, 50))
    features = np.random.uniform(0, 1, (32, 50, 100))
    graph = GraphLayer(output_dim=96)([mask, adj, features])
    output = ReadoutLayer(output_dim=96)([mask, graph])
    print(output.shape)

