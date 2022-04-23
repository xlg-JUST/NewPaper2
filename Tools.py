from tensorflow.keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


def str2list(texts):
    """
    cut the texts into words list
    :param texts: str sentences
    :return: words list
    """
    sentences = []
    for sen in texts:
        if type(sen) != float:
            sentences.append(sen.strip().split())
        else:  # 遇到空句子，值为nan float型
            sentences.append([0])
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
    return pad_sequences(sequences, maxlen=maxlen, padding='post', value=0)


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
        precision, recall, specificity = tp/(tp+fp+1e-4), tp/(tp+fn+1e-4), tn/(tn+fp+1e-4)
        f1, gmean = 2*precision*recall/(precision+recall+1e-4), np.sqrt(recall*specificity)
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
    testdata, testlabels = [], []
    for file in trainfiles:
        if file == target:
            continue
        df = pd.read_csv(filepath+file)
        comments, labels = df['preprocess_comments'].to_list(), df['classification'].to_list()
        for i in range(len(comments)):
            if type(comments[i]) != float:
                traindata.append(comments[i])
                trainlabels.append(labels[i])
            else:
                continue  # skip nan

    df = pd.read_csv(filepath+target)
    comments, labels = df['preprocess_comments'].to_list(), df['classification'].to_list()
    for i in range(len(comments)):
        if type(comments[i]) != float:
            testdata.append(comments[i])
            testlabels.append(labels[i])
        else:
            continue  # skip nan
    return traindata, testdata, np.array(trainlabels), np.array(testlabels)


def within_project(filepath, testsize):
    """
    训练集为list，测试集为dict（每个项目名为key）
    :param filepath:
    :param testsize:
    :return:
    """
    files = os.listdir(filepath)
    train_data, train_labels = [], []
    test_data, test_labels = {}, {}
    for file in files:
        df = pd.read_csv(filepath+file)
        cur_comments, cur_labels = df['preprocess_comments'].to_list(), df['classification'].to_list()
        comments, labels = [], []
        for i in range(len(cur_comments)):
            if type(cur_comments[i]) != float:
                comments.append(cur_comments[i])
                labels.append(cur_labels[i])
            else:
                continue  # skip nan
        x_train, x_test, y_train, y_test = train_test_split(comments, labels, test_size=testsize, random_state=1)
        train_data.extend(x_train), train_labels.extend(y_train)
        test_data[file] = x_test
        test_labels[file] = np.array(y_test)
    return train_data, np.array(train_labels), test_data, test_labels


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
        cur_comments, cur_label = df['preprocess_comments'].to_list(), df['classification'].to_list()
        for i in range(len(cur_comments)):
            if type(cur_comments[i]) != float:
                comments.append(cur_comments[i])
                labels.append(cur_label[i])
            else:
                continue  # skip nan
    x_train, x_test, y_train, y_test = train_test_split(comments, labels, test_size=testsize)
    return x_train, x_test, np.array(y_train), np.array(y_test)


def index2gram(sentences, n=2):
    """
    NgramIDF使用，将index转换为gram形式
    :param sentences:
    :return:
    """
    grams = []
    for sen in sentences:
        x = []
        temp = [str(idx) for idx in sen]
        for tmp in zip(*[temp[i:] for i in range(n)]):
            tmp = '_'.join(tmp)
            x.append(tmp)
        grams.append(x)
    return grams


def build_bow(sentences, vocab, iflist=True):
    """
    CountVectorizer 只接受一维向量内含string，因此如果是已经切词完成的list，需要重新组合为string
    :param sentences:
    :param vocab:
    :param iflist:
    :return:
    """
    cv = CountVectorizer(stop_words=None, vocabulary=vocab)
    if iflist is False:
        nan = np.argwhere(pd.isnull(sentences)).reshape(-1)  # 处理NAN
        for idx in nan:
            sentences[idx] = '[PAD]'
    if iflist:
        nan = np.argwhere(pd.isnull(sentences)).reshape(-1)  # 处理NAN
        for idx in nan:
            sentences[idx].append('[PAD]')
        sentences = [' '.join('%s' % idx for idx in sen) for sen in sentences]
    return cv.fit_transform(sentences).toarray()


if __name__ == '__main__':
    pass