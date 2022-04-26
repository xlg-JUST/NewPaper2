import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding, LSTM
from tensorflow.keras.layers import Dot, Bidirectional, Dropout
from Models.Ours.origin_Attention import Attention
from Tools import *
import os


class LSTMAtt:
    def __init__(self, sen_len, vocab_size, wvdim, hidden_size, num_classes=3):

        inputs = Input(shape=(sen_len, ))
        embedding = Embedding(vocab_size, wvdim, input_length=sen_len, mask_zero=True)(inputs)
        hidden = Bidirectional(LSTM(hidden_size, return_sequences=True))(embedding)
        attention = Attention(step_dim=hidden.shape[1])(hidden)
        outputs = Dense(num_classes, activation='softmax')(attention)

        self.model = Model(inputs, outputs)
        self.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam')

    def fit(self, x_train, y_train, epoch):
        self.model.fit(x_train, y_train, epochs=epoch)
        return 0

    def predict(self, x_test):
        """
        :param X:
        :return: outputs:类别概率分布，用于计算多类AUC
                 predictions：最终分类类别，用于计算g-mean，precision，recall，f1
        """
        score = self.model.predict(x_test)  # for AUC
        predictions = np.argmax(score, axis=1)  # for else
        return predictions, score


if __name__ == '__main__':
    """
    准备数据
    """
    vocab = open(r'../../Dataset/Vocab/min_count=5.txt', encoding='utf-8').readline().strip().split()
    filepath = '../../Dataset/SelectedData/'
    save_path = r'../../Experiment Results/LSTM_Att/'
    files = os.listdir(filepath)

    ## 跨项目
    # design, requirement = [], []
    # label_names = ['Non-SATD', 'Design', 'Requirement']
    # metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    # for file in files:
    #     x_train, x_test, y_train, y_test = cross_project(filepath, file)
    #     x_train, x_test = str2list(x_train), str2list(x_test)
    #     x_train, x_test = word2index(x_train, vocab), word2index(x_test, vocab)
    #     x_train, x_test = pad(x_train, maxlen=50), pad(x_test, maxlen=50)
    #     cur_model = LSTMAtt(sen_len=50, vocab_size=len(vocab), wvdim=100, hidden_size=50)
    #     cur_model.fit(x_train, y_train, epoch=3)
    #     y_pred, y_score = cur_model.predict(x_test)
    #     result = performence(y_test, y_pred, y_score, label_names)
    #
    #     design.append(result[1, :].reshape(-1)), requirement.append(result[2, :].reshape(-1))
    # design, requirement = np.vstack(design), np.vstack(requirement)
    # de_df = pd.DataFrame(design, index=files, columns=metric)
    # re_df = pd.DataFrame(requirement, index=files, columns=metric)
    # de_df.to_csv(save_path + 'CrossProject_Design.csv')
    # re_df.to_csv(save_path + 'CrossProject_Requirement.csv')

    # within
    design, requirement = [], []
    label_names = ['Non-SATD', 'Design', 'Requirement']
    metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    train_data, train_labels, test_data, test_labels = within_project(filepath, testsize=0.3, random_state=1)
    train_data = pad(word2index(str2list(train_data), vocab), 50)
    cur_model = LSTMAtt(sen_len=50, vocab_size=len(vocab), wvdim=100, hidden_size=50)
    cur_model.fit(train_data, train_labels, epoch=5)
    for key in test_data.keys():
        x_test, y_test = test_data[key], test_labels[key]
        x_test = pad(word2index(str2list(x_test), vocab), 50)
        y_pred, y_score = cur_model.predict(x_test)
        result = performence(y_test, y_pred, y_score, label_names)
        design.append(result[1, :].reshape(-1)), requirement.append(result[2, :].reshape(-1))
    de_df = pd.DataFrame(design, index=files, columns=metric)
    re_df = pd.DataFrame(requirement, index=files, columns=metric)
    de_df.to_csv(save_path + 'WithinProject_Design.csv')
    re_df.to_csv(save_path + 'WithinProject_Requirement.csv')

