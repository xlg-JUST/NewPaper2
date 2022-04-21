import numpy as np
import pickle as pkl

import pandas as pd
import scipy.sparse as sp
# from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import random
import re
from tqdm import tqdm
import os


def load_data(file):

    with open("data/{}---{}".format(file, 'x_adj'), 'rb') as f:
        x_adj = pkl.load(f, encoding='latin1')
    with open("data/{}---{}".format(file, 'x_features'), 'rb') as f:
        x_features = pkl.load(f, encoding='latin1')
    with open("data/{}---{}".format(file, 'labels'), 'rb') as f:
        labels = pkl.load(f, encoding='latin1')

    # x_adj = np.array([elm.toarray() for elm in x_adj])  内存炸了
    return x_adj, x_features, labels


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
    max_length = 300

    for i in range(len(features)):
        feature = np.array(features[i])
        if feature.shape[0] >= max_length:
            feature = feature[: max_length, :]
        else:
            pad = max_length - feature.shape[0]  # padding for each epoch
            feature = np.pad(feature, ((0, pad), (0, 0)), mode='constant')
        features[i] = feature

    return np.array(features)


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

    max_length = 300
    mask = np.zeros((len(adj), max_length, 1))  # mask for padding
    adj = [elm.toarray() for elm in adj]  # 展开为array，配合生成器使用
    for i in range(len(adj)):
        if adj[i].shape[0] < max_length:
            adj_normalized = normalize_adj(adj[i])  # no self-loop
            pad = max_length - adj_normalized.shape[0] # padding for each epoch
            adj_normalized = np.pad(adj_normalized, ((0,pad),(0,pad)), mode='constant')
            mask[i, :adj[i].shape[0], :] = 1.
            adj[i] = adj_normalized
        else:
            adj_normalized = normalize_adj(adj[i][:max_length, :max_length])
            mask[i, :, :] = 1.
            adj[i] = adj_normalized
    return np.array(adj), mask  # coo_to_tuple(sparse.COO(np.array(list(adj)))), mask


def construct_feed_dict(features, support, mask, labels, placeholders):
    """Construct feed dictionary."""
    """support其实就是x_adj"""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support']: support})
    feed_dict.update({placeholders['mask']: mask})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def cross_project(tar_file):
    train_adj, train_features, train_labels = [], [], []
    files = os.listdir(r'../../Dataset/SelectedData')
    for file in files:
        if file == '.DS_Store':
            continue
        if file == tar_file:
            continue
        x_adj, x_features, labels = load_data(file)
        train_adj.extend(x_adj)
        train_features.extend(x_features)
        train_labels.extend(labels)
    test_adj, test_features, test_labels = load_data(tar_file)
    return train_adj, train_features, train_labels, test_adj, test_features, test_labels


if __name__ == '__main__':
    pass
