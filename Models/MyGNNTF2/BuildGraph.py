import os
import pandas as pd
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
from gensim.models import Word2Vec

total_vocab = open(r'../MyGNNTF2/data/total_vocab.txt').readline().strip().split()
word_id_map = {}
for i in range(len(total_vocab)):
    word_id_map[total_vocab[i]] = i
w2v_vocab = open(r'../MyGNNTF2/data/vocab.txt').readline().strip().split()


def build_graph(filepath, file):
    """
    使用总词库、mincount词库建立graph
    :param file:
    :return:
    """
    window_size = 6
    x_adj = []
    x_feature = []
    df = pd.read_csv(filepath+file, encoding='windows-1252')
    tmp_comments = df['preprocess_comments'].to_list()
    tmp_labels = df['classification'].to_list()

    """Skip NAN"""
    comments, labels = [], []
    for i in range(len(tmp_comments)):
        if type(tmp_comments[i]) != float:
            comments.append(tmp_comments[i])  # str,用extend就直接切割开了
            labels.append(tmp_labels[i])
        else:
            continue  # skip nan

    for i in range(len(comments)):
        doc_words = comments[i].strip().split()
        doc_len = len(doc_words)
        doc_vocab = list(set(doc_words))
        doc_nodes = len(doc_vocab)
        doc_word_id_map = {}
        for j in range(doc_nodes):
            doc_word_id_map[doc_vocab[j]] = j

        windows = []
        if doc_len <= window_size:
            windows.append(doc_words)
        else:
            for j in range(doc_len - window_size + 1):
                window = doc_words[j: j + window_size]
                windows.append(window)
        word_pair_count = {}
        for window in windows:
            for p in range(1, len(window)):
                for q in range(0, p):
                    word_p = window[p]
                    word_p_id = word_id_map[word_p]
                    word_q = window[q]
                    word_q_id = word_id_map[word_q]
                    if word_p_id == word_q_id:
                        continue
                    word_pair_key = (word_p_id, word_q_id)
                    # word co-occurrences as weights
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.
                    # bi-direction
                    word_pair_key = (word_q_id, word_p_id)
                    if word_pair_key in word_pair_count:
                        word_pair_count[word_pair_key] += 1.
                    else:
                        word_pair_count[word_pair_key] = 1.

        row = []
        col = []
        weight = []
        features = []

        for key in word_pair_count:
            p = key[0]
            q = key[1]
            row.append(doc_word_id_map[total_vocab[p]])
            col.append(doc_word_id_map[total_vocab[q]])
            weight.append(word_pair_count[key])
        adj = sp.csr_matrix((weight, (row, col)), shape=(doc_nodes, doc_nodes))
        for k, v in sorted(doc_word_id_map.items(), key=lambda x: x[1]):
            features.append(w2v_vocab.index(k) if k in w2v_vocab else 0)

        x_adj.append(adj)
        x_feature.append(features)

    with open(r'data/'+file+'---x_adj', 'wb',) as f:
        pkl.dump(x_adj, f)
    with open(r'data/'+file+'---x_features', 'wb',) as f:
        pkl.dump(x_feature, f)
    with open(r'data/'+file+'---labels', 'wb',) as f:
        pkl.dump(labels, f)
    return 0


if __name__ == '__main__':
    filepath = '../../Dataset/SelectedData/'
    files = os.listdir(filepath)
    # # 建立W2vmodel 和mincount=5 的词表
    # sentences = []
    # for file in files:
    #     if file == '.DS_Store':
    #         continue
    #     df = pd.read_csv(filepath+file, encoding='windows-1252')
    #     comments = df['preprocess_comments'].to_list()
    #     for sen in comments:
    #         if type(sen) != float:
    #             sentences.append(sen.strip().split())
    #         else:  # 遇到空句子，值为nan float型
    #             continue
    # w2vmodel = Word2Vec(sentences, min_count=5, vector_size=100)
    # vocab = list(w2vmodel.wv.key_to_index.keys())
    # vocab.insert(0, '[PAD]')
    # w2vmodel.save(r'../../Models/MyGNNTF2/data/W2Vmodel.model')
    # f = open(r'../../Models/MyGNNTF2/data/vocab.txt', 'w', encoding='utf-8')
    # for word in vocab:
    #     f.write(word+' ')
    # f.close()
    #
    # # 建立总词表
    # vocab = set()
    # for file in files:
    #     if file == '.DS_Store':
    #         continue
    #     df = pd.read_csv(filepath+file, encoding='windows-1252')
    #     comments = df['preprocess_comments'].to_list()
    #     for sen in comments:
    #         if type(sen) != float:
    #             vocab.update(sen.strip().split())
    #         else:
    #             continue
    # vocab = list(vocab)
    # vocab.insert(0, '[PAD]')
    # f = open(r'../MyGNNTF2/data/total_vocab.txt', 'w')
    # for word in vocab:
    #     f.write(word+' ')
    # f.close()
    for file in files:
        if file == '.DS_Store':
            continue
        build_graph(filepath, file)

