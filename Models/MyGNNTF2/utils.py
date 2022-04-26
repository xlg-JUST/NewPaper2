import numpy as np
import pickle as pkl
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import scipy.sparse as sp
import tensorflow as tf
import os

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


def load_data(file):

    with open("data/{}---{}".format(file, 'x_adj'), 'rb') as f:
        x_adj = pkl.load(f, encoding='latin1')
    with open("data/{}---{}".format(file, 'x_features'), 'rb') as f:
        x_features = pkl.load(f, encoding='latin1')
    with open("data/{}---{}".format(file, 'labels'), 'rb') as f:
        labels = pkl.load(f, encoding='latin1')

    # x_adj = np.array([elm.toarray() for elm in x_adj])  内存炸了
    return x_adj, x_features, np.array(labels)


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    return pad_sequences(features, maxlen=FLAGS.graph_len, padding='post', value=0)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    with np.errstate(divide='ignore'):
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""

    mask = np.zeros((len(adj), FLAGS.graph_len, 1))  # mask for padding
    adj = [elm.toarray() for elm in adj]  # 展开为array，配合生成器使用
    for i in range(len(adj)):
        if adj[i].shape[0] < FLAGS.graph_len:
            adj_normalized = normalize_adj(adj[i])  # no self-loop
            pad = FLAGS.graph_len - adj_normalized.shape[0] # padding for each epoch
            adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
            mask[i, :adj[i].shape[0], :] = 1.
            adj[i] = adj_normalized
        else:
            adj_normalized = normalize_adj(adj[i][:FLAGS.graph_len, :FLAGS.graph_len])
            mask[i, :, :] = 1.
            adj[i] = adj_normalized
    return np.array(adj), mask  # coo_to_tuple(sparse.COO(np.array(list(adj)))), mask


def construct_input_sequence(mask, adj, features):
    features = np.expand_dims(features, -1)
    return np.concatenate([mask, adj, features], axis=-1)


def cross_project(filepath, target):
    files = os.listdir(filepath)
    train_input_sequence, train_labels = [], []  # list
    for file in files:
        if file == target:
            continue
        adj, features, labels = load_data(file)
        adj, mask = preprocess_adj(adj)
        features = preprocess_features(features)
        input_sequence = construct_input_sequence(mask, adj, features)
        train_input_sequence.extend(input_sequence), train_labels.extend(labels)
    adj, features, labels = load_data(target)
    adj, mask = preprocess_adj(adj)
    features = preprocess_features(features)
    test_input_sequence, test_labels = construct_input_sequence(mask, adj, features), labels
    return np.array(train_input_sequence), test_input_sequence, np.array(train_labels), test_labels


def within_project(filepath, testsize):
    files = os.listdir(filepath)
    train_input_sequence, train_labels = [], []  # list
    test_input_sequence, test_labels = {}, {}  # dict
    for file in files:
        adj, features, labels = load_data(file)
        adj, mask = preprocess_adj(adj)
        features = preprocess_features(features)
        input_sequence = construct_input_sequence(mask, adj, features)
        x_train, x_test, y_train, y_test = train_test_split(input_sequence, labels, test_size=testsize)
        train_input_sequence.extend(x_train), train_labels.extend(y_train)
        test_input_sequence[file] = np.array(x_test)
        test_labels[file] = np.array(y_test)
    return np.array(train_input_sequence), test_input_sequence, np.array(train_labels), test_labels


def mix_project(filepath, testsize):
    files = os.listdir(filepath)
    input_sequences, total_labels = [], []
    for file in files:
        adj, features, labels = load_data(file)
        adj, mask = preprocess_adj(adj)
        features = preprocess_features(features)
        input_sequences.extend(construct_input_sequence(mask, adj, features))
        total_labels.extend(labels)
    input_sequences, total_labels = np.array(input_sequences), np.array(total_labels)
    x_train, x_test, y_train, y_test = train_test_split(input_sequences, total_labels, test_size=testsize, random_state=1)
    return x_train, x_test, y_train, y_test


if __name__ == '__main__':
    pass