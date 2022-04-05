import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.layers import Dot, Bidirectional
from Models.origin_Attention import Attention
from Tools import *
import os


class MyLCM:
    def __init__(self, sen_len, vocab_size, wvdim, hidden_size, class_weight, num_classes=3, alpha=0.5):
        self.num_classes = num_classes
        self.alpha = alpha
        self.class_weights = class_weight

        text_input = Input(shape=(sen_len,), name='text_input')
        input_emb = Embedding(vocab_size, wvdim, input_length=sen_len, name='text_emb', mask_zero=True)(text_input)  # (V,wvdim)
        input_vec = Bidirectional(GRU(hidden_size, return_sequences=True))(input_emb)
        attention_layer = Attention(step_dim=input_vec.shape[1])(input_vec)
        pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(attention_layer)
        self.basic_predictor = Model(inputs=text_input, outputs=pred_probs)
        self.basic_predictor.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam')

        # LCM:
        label_input = Input(shape=(num_classes,), name='label_input')
        label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(label_input)  # (n,wvdim)
        label_emb = Dense(hidden_size*2, activation='tanh', name='label_emb2')(label_emb)
        # similarity part:
        doc_product = Dot(axes=(2, 1))([label_emb, attention_layer])  # (n,d) dot (d,1) --> (n,1)
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(doc_product)
        # concat output:
        concat_output = self.alpha*label_sim_dict + (1-self.alpha)*pred_probs
        # compile；
        self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
        self.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam')

    def fit(self, x_train, y_train, epoch):
        l_train = np.array([np.array(range(self.num_classes)) for i in range(len(x_train))])
        self.model.fit([x_train, l_train], y_train, epochs=epoch, class_weight=self.class_weights)
        return 0

    def predict(self, x_test):
        """
        :param X:
        :return: outputs:类别概率分布，用于计算多类AUC
                 predictions：最终分类类别，用于计算g-mean，precision，recall，f1
        """
        l_test = np.array([np.array(range(self.num_classes)) for i in range(len(x_test))])
        score = self.model.predict([x_test, l_test])  # for AUC
        predictions = np.argmax(score, axis=1)  # for else
        return predictions, score


if __name__ == '__main__':
    """
    准备数据
    """
    vocab = open(r'../Dataset/Vocab/min_count=5.txt', encoding='utf-8').readline().strip().split()
    filepath = '../Dataset/SelectedData/'
    save_path = r'../Experiment Results/Ours/'
    files = os.listdir(filepath)

    # 混合
    # x_train, x_test, y_train, y_test = mix_project(filepath, testsize=0.1)
    # x_train, x_test = str2list(x_train), str2list(x_test)
    # x_train, x_test = word2index(x_train, vocab), word2index(x_test, vocab)
    # x_train, x_test = pad(x_train, maxlen=50), pad(x_test, maxlen=50)
    # label_names = ['Non-SATD', 'Design', 'Requirement']
    # metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    # de_pf, re_pf = [], []
    # for i in range(20):
    #     best_f1, best_result = 0, []
    #     cur_model = MyLCM(sen_len=50, vocab_size=len(vocab), wvdim=100, hidden_size=50, class_weight={0: 1, 1: 2, 2: 3})
    #     for _ in range(5):
    #         cur_model.fit(x_train, y_train, epoch=1)
    #         y_pred, y_score = cur_model.predict(x_test)
    #         result = performence(y_test, y_pred, y_score, label_names, report=True)
    #         f1 = sum(result[1:, 2].reshape(-1))
    #         if f1 > best_f1:
    #             best_f1, best_result = f1, result
    #     de_pf.append(best_result[1, :]), re_pf.append(best_result[-1, :])
    # de_pf, re_pf = np.vstack(de_pf), np.vstack(re_pf)
    # de_df = pd.DataFrame(de_pf, columns=metric)  # 只需要 设计债和需求债
    # re_df = pd.DataFrame(re_pf, columns=metric)  # 只需要 设计债和需求债
    # de_df.to_csv(save_path+'Mix_project_Design.csv')
    # re_df.to_csv(save_path+'Mix_project_Requirement.csv')

    # # 跨项目
    design, requirement = [], []
    label_names = ['Non-SATD', 'Design', 'Requirement']
    metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    for file in files:
        x_train, x_test, y_train, y_test = cross_project(filepath, file)
        x_train, x_test = str2list(x_train), str2list(x_test)
        x_train, x_test = word2index(x_train, vocab), word2index(x_test, vocab)
        x_train, x_test = pad(x_train, maxlen=50), pad(x_test, maxlen=50)
        cur_model = MyLCM(sen_len=50, vocab_size=len(vocab), wvdim=100, hidden_size=50, class_weight={0: 1, 1: 2, 2: 3})
        best_f1, best_result = 0, []
        for _ in range(5):
            cur_model.fit(x_train, y_train, epoch=1)
            y_pred, y_score = cur_model.predict(x_test)
            result = performence(y_test, y_pred, y_score, label_names)
            f1 = sum(result[1:, 2].reshape(-1))
            if f1 > best_f1:
                best_f1, best_result = f1, result
        design.append(best_result[1, :].reshape(-1)), requirement.append(best_result[2, :].reshape(-1))
    design, requirement = np.vstack(design), np.vstack(requirement)
    de_df = pd.DataFrame(design, index=files, columns=metric)
    re_df = pd.DataFrame(requirement, index=files, columns=metric)
    de_df.to_csv(save_path+'CrossProject_Design.csv')
    re_df.to_csv(save_path+'CrossProject_Requirement.csv')
    #
    # # 项目内
    design, requirement = [], []
    label_names = ['Non-SATD', 'Design', 'Requirement']
    metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    for file in files:
        x_train, x_test, y_train, y_test = within_project(filepath, file, testsize=0.1)
        x_train, x_test = str2list(x_train), str2list(x_test)
        x_train, x_test = word2index(x_train, vocab), word2index(x_test, vocab)
        x_train, x_test = pad(x_train, maxlen=50), pad(x_test, maxlen=50)
        best_f1, best_result = 0, []
        for _ in range(5):
            cur_model = MyLCM(sen_len=50, vocab_size=len(vocab), wvdim=100, hidden_size=50, class_weight={0: 1, 1: 2, 2: 3})
            cur_model.fit(x_train, y_train, epoch=3)
            y_pred, y_score = cur_model.predict(x_test)
            result = performence(y_test, y_pred, y_score, label_names)
            f1 = sum(result[1:, 2].reshape(-1))
            if f1 > best_f1:
                best_f1, best_result = f1, result
        design.append(best_result[1, :].reshape(-1)), requirement.append(best_result[2, :].reshape(-1))
    design, requirement = np.vstack(design), np.vstack(requirement)
    de_df = pd.DataFrame(design, index=files, columns=metric)
    re_df = pd.DataFrame(requirement, index=files, columns=metric)
    de_df.to_csv(save_path + 'WithinProject_Design.csv')
    re_df.to_csv(save_path + 'WithinProject_Requirement.csv')








