import time
import tensorflow as tf
from sklearn import metrics
import pickle as pkl

from utils import *
from models import GNN, MLP


flags = tf.compat.v1.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('dataset', 'mr', 'Dataset string.')  # 'mr','ohsumed','R8','R52'
flags.DEFINE_string('model', 'gnn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 5, 'Number of epochs to train.')
flags.DEFINE_integer('batch_size', 32, 'Size of batches per epoch.')
flags.DEFINE_integer('input_dim', 100, 'Dimension of input.')
flags.DEFINE_integer('hidden', 50, 'Number of units in hidden layer.') # 32, 64, 96, 128
flags.DEFINE_integer('steps', 2, 'Number of graph layers.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0, 'Weight for L2 loss on embedding matrix.') # 5e-4
flags.DEFINE_integer('early_stopping', -1, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') # Not used

"""
测试一下
"""
file = 'emf-2.4.1.csv'
x_adj, x_features, labels = load_data(file)
labels = np.eye(3)[labels]  # one-hot
batch_size = 32

placeholders = {
    'support': tf.compat.v1.placeholder(tf.float32, shape=(None, None, None)),
    'features': tf.compat.v1.placeholder(tf.float32, shape=(None, None, FLAGS.input_dim)),
    'mask': tf.compat.v1.placeholder(tf.float32, shape=(None, None, 1)),
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, labels.shape[1])),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
}


def evaluate(features, support, mask, labels, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, mask, labels, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.embeddings, model.preds, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test), outs_val[2], outs_val[3], outs_val[4]


model = GNN(placeholders, input_dim=100, logging=False, )


sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())

train_loss, train_acc = 0, 0
for start in range(0, len(labels), batch_size):
    end = start + FLAGS.batch_size
    train_adj, train_mask = preprocess_adj(x_adj[start:end])
    train_features = preprocess_features(x_features[start:end])
    train_labels = labels[start:end]

    # Construct feed dictionary
    feed_dict = construct_feed_dict(train_features, train_adj, train_mask, train_labels, placeholders)
    feed_dict.update({placeholders['dropout']: 0.5})

    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    train_loss += outs[1]*batch_size
    train_acc += outs[2]*batch_size
train_loss /= len(labels)
train_acc /= len(labels)





