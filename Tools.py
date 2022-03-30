from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def str2list(texts):
    """
    cut the texts into words list
    :param texts: str sentences
    :return: words list
    """
    sentences = []
    for sen in texts:
        sentences.append(sen.strip().split())
    return sentences


def build_vocab(sentences, min_count):
    """
    并不是要训练W2V模型，只是它的去低频词功能太方便了。。。
    :param sentences:
    :param min_count:
    :return:
    """
    model = Word2Vec(sentences=sentences, min_count=min_count, vector_size=10, window=5, workers=4)
    vocab = list(model.wv.key_to_index.keys())
    vocab.insert(0, '')
    return vocab


def word2index(sentences, vocab):
    idx = []
    for sen in sentences:
        idx.append([vocab.index(word) for word in sen if word in vocab])
    return idx


def pad(sequences, maxlen):
    return pad_sequences(sequences, maxlen=maxlen, value=0)


def performence(y_true, y_pred, y_score, label_names, report=False, save_path=None):
    """

    :param y_true:
    :param y_pred:
    :param y_score:
    :param label_names:
    :param report:
    :param save_path:
    :return:
    """
    pre_ls, re_ls, f1_ls, gm_ls, auc_ls = [], [], [], [], []
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    labelcount = mcm.shape[0]
    assert len(label_names) == labelcount, '请核对标签种类的数量'
    onehot_ytrue = np.eye(labelcount)[y_true]
    for i in range(labelcount):
        cur_cm, cur_onehot_ytrue, cur_yscore = mcm[i], onehot_ytrue[:, i].reshape(-1), y_score[:, i].reshape(-1)
        tn, fp, fn, tp = cur_cm[0, 0], cur_cm[0, 1], cur_cm[1, 0], cur_cm[1, 1]
        precision, recall, specificity = tp/(tp+fp), tp/(tp+fn), tn/(tn+fp)
        f1, gmean = 2*precision*recall/(precision+recall), np.sqrt(recall*specificity)
        auc = roc_auc_score(y_true=cur_onehot_ytrue, y_score=cur_yscore)
        pre_ls.append(precision), re_ls.append(recall), f1_ls.append(f1), gm_ls.append(gmean), auc_ls.append(auc)
        if report:
            print('Label:{}'.format(label_names[i]))
            print('Precision:{}%'.format(np.around(precision, decimals=4)*100))
            print('Recall:   {}%'.format(np.around(recall, decimals=4)*100))
            print('F1_score: {}%'.format(np.around(f1, decimals=4)*100))
            print('G_mean:   {}%'.format(np.around(gmean, decimals=4)*100))
            print('AUC:      {}%'.format(np.around(auc, decimals=4)*100))
        if save_path is not None:
            performence_ls = {'Precision': pre_ls, 'Recall': re_ls, 'F1_score': f1_ls, 'G_mean': gm_ls, 'AUC': auc_ls}
            df = pd.DataFrame(data=performence_ls, index=label_names)
            df.to_csv(save_path)
    # pre_ls, re_ls, f1_ls, gm_ls, auc_ls = np.array(pre_ls), np.array(re_ls), np.array(f1_ls), np.array(gm_ls), np.array(auc_ls)
    performence = np.vstack([pre_ls, re_ls, f1_ls, gm_ls, auc_ls])
    return performence.T


def cross_project(filepath, target):
    """

    :param filepath: 数据集路径
    :param target: 测试集，除target以外的数据集用作训练集
    :return:
    """
    trainfiles = os.listdir(filepath)
    assert target in trainfiles, '目标未在数据集中，请核实'
    traindata, trainlabels = [], []
    for file in trainfiles:
        if file == target:
            continue
        df = pd.read_csv(filepath+file)
        comments, labels = df['preprocess_comments'].to_list(), df['classification'].to_list()
        traindata.extend(comments), trainlabels.extend(labels)
    df = pd.read_csv(filepath+target)
    testdata, testlabels = df['preprocess_comments'].to_list(), df['classification'].to_list()

    return traindata, testdata, trainlabels, testlabels


def within_project(filepath, target, testsize):
    """
    项目内预测
    :param filepath:
    :param target:
    :param testsize:
    :return:
    """
    df = pd.read_csv(filepath+target)
    comments, labels = df['preprocess_comments'].to_list(), df['classification'].to_list()
    return train_test_split(comments, labels, test_size=testsize, random_state=1)


def mix_project(filepath, testsize):
    """
    混合项目预测
    :param filepath:
    :param testsize:
    :return:
    """
    files = os.listdir(filepath)
    comments, labels = [], []
    for file in files:
        df = pd.read_csv(filepath+file)
        comment, label = df['preprocess_comments'].to_list(), df['classification'].to_list()
        comments.extend(comment), labels.extend(label)
    return train_test_split(comments, labels, test_size=testsize, random_state=1)








