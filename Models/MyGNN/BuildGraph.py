import os
import pandas as pd
import random
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import sys
from tqdm import tqdm
from gensim.models import Word2Vec


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
filepath = '../../Dataset/SelectedData/'
save_path = r'../../Experiment Results/GNN/'
files = os.listdir(filepath)
W2vModel = Word2Vec.load(r'data/W2Vmodel.model')
vocab = open(r'data/vocab.txt', encoding='utf-8').readline().strip().split()
vocab.insert(0, '[PAD]')
pad_vec = np.array([0.0]*100)  # 用于填充nan


def build_graph(filepath, file):
    window_size = 6
    x_adj = []
    x_feature = []
    df = pd.read_csv(filepath+file, encoding='windows-1252')
    comments = df['preprocess_comments'].to_list()
    labels = df['classification'].to_list()
    """
    features 这部分写的很烂，因为是一次性工作所以无所谓了，后期有需要的时候再改吧
    """
    for i in range(len(comments)):
        if type(comments[i]) == float:  # nan
            x_feature.append(np.array([pad_vec]))
            comments[i] = ['PAD']  # 替换nan
        else:
            comments[i] = comments[i].strip().split()  # 将str切分为词，方便后续操作
            comments[i] = [word for word in comments[i] if word in vocab]  # 这一步有时候导致出现空列表 因此后面有if语句
            tmp = [W2vModel.wv[word] for word in comments[i]]
            if tmp == []:
                tmp = np.array([pad_vec])
            x_feature.append(np.array(tmp))

    for sen in comments:
        if len(sen) == 1:
            tmp = np.array([np.array([0])])
            x_adj.append(sp.csr_matrix(tmp, shape=(1, 1)))

        else:
            windows = []
            sen_vocab = list(set(sen))
            nodes = len(sen_vocab)  # 一句注释有多少个词，那么该注释的图就有多少个节点，跟词库大小无关
            if len(sen) <= window_size:
                windows.append(sen)
            else:
                for j in range(len(sen) - window_size + 1):
                    window = sen[j: j + window_size]
                    windows.append(window)

            word_pair_count = {}
            for window in windows:
                for p in range(1, len(window)):
                    for q in range(0, p):
                        word_p = window[p]
                        word_p_id = vocab.index(word_p)
                        word_q = window[q]
                        word_q_id = vocab.index(word_q)
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

            for key in word_pair_count:
                p = key[0]
                q = key[1]
                row.append(sen_vocab.index(vocab[p]))
                col.append(sen_vocab.index(vocab[q]))
                weight.append(word_pair_count[key])
            adj = sp.csr_matrix((weight, (row, col)), shape=(nodes, nodes))
            x_adj.append(adj)

    with open(r'data/'+file+'---x_adj', 'wb',) as f:
        pkl.dump(x_adj, f)
    with open(r'data/'+file+'---x_features', 'wb',) as f:
        pkl.dump(x_feature, f)
    with open(r'data/'+file+'---labels', 'wb',) as f:
        pkl.dump(labels, f)
    return 0


if __name__ == '__main__':
    # for file in files:
    #     if file == '.DS_Store':
    #         continue
    #     build_graph(filepath, file)
    pass




