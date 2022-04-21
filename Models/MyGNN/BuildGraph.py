import os
import pandas as pd
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
from gensim.models import Word2Vec

filepath = '../../Dataset/SelectedData/'
files = os.listdir(filepath)
save_path = r'../../Experiment Results/GNN/'
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
# vocab = tuple(w2vmodel.wv.key_to_index.keys())
# w2vmodel.save(r'../../Models/MyGNN/W2Vmodel.model')
# f = open(r'../../Models/MyGNN/vocab.txt', 'w', encoding='utf-8')
# for word in vocab:
#     f.write(word+' ')
# f.close()

# total_vocab = set()
# for file in files:
#     if file == '.DS_Store':
#         continue
#     df = pd.read_csv(filepath + file, encoding='windows-1252')
#     comments = df['preprocess_comments'].to_list()
#     for sen in comments:
#         if type(sen) != float:
#             words = sen.strip().split()
#             total_vocab.update(words)
#         else:
#             continue
# total_vocab = list(total_vocab)
# f = open(r'data/total_vocab.txt', 'w', encoding='utf-8')
# for word in total_vocab:
#     f.write(word + ' ')
# f.close()

total_vocab = open(r'data/total_vocab.txt', encoding='utf-8').readline().strip().split()
W2vModel = Word2Vec.load(r'data/W2Vmodel300.model')
embedding_vocab = open(r'data/vocab.txt', encoding='utf-8').readline().strip().split()

word_id_map = {}
for i in range(len(total_vocab)):
    word_id_map[total_vocab[i]] = i

oov = {}
for v in total_vocab:
    oov[v] = np.random.uniform(-0.01, 0.01, 300)


# vocab.insert(0, '[PAD]')
# pad_vec = np.array([0.0]*100)  # 用于填充nan


def build_graph(filepath, file):
    window_size = 6
    x_adj = []
    x_feature = []
    df = pd.read_csv(filepath+file, encoding='windows-1252')
    tmp_comments = df['preprocess_comments'].to_list()
    tmp_labels = df['classification'].to_list()
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
            features.append(W2vModel.wv[k] if k in embedding_vocab else oov[k])

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
    for file in files:
        if file == '.DS_Store':
            continue
        build_graph(filepath, file)


