from Models.MyGNNTF2.utils import *
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Embedding, Dense
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from Models.MyGNNTF2.layers import *
import tensorflow as tf
from Tools import performence
import pandas as pd

"""
参数：graph_len 图大小，即构成图的句子的最大长度
     features_dim 词向量长度
     vocab_size 词库大小（未包含pad，因此embedding中size+1）
     hidden GGNN的隐藏层大小
"""


class GGNN:
    def __init__(self, graph_len, features_dim, vocab_size, hidden):
        self.graph_len = graph_len
        self.features_dim = features_dim
        self.vocab_size = vocab_size
        self.hidden = hidden

        input_sequence = Input(shape=(self.graph_len, self.graph_len+2))
        mask = K.expand_dims(input_sequence[:, :, 0], -1)
        adj = input_sequence[:, :, 1:-1]
        features = input_sequence[:, :, -1]
        embedding_layer = Embedding(input_dim=self.vocab_size+1, output_dim=self.features_dim, mask_zero=True)(features)
        graph_layer1 = GraphLayer(input_dim=self.features_dim, output_dim=self.hidden)([mask, adj, embedding_layer])
        graph_layer2 = GraphLayer(input_dim=self.hidden, output_dim=self.hidden)([mask, adj, graph_layer1])
        readout_layer = ReadoutLayer(output_dim=self.hidden)([mask, graph_layer2])
        output_layer = Dense(3, activation='softmax')(readout_layer)

        self.model = Model(inputs=input_sequence, outputs=output_layer)
        self.model.compile(loss=SparseCategoricalCrossentropy(), optimizer='adam')

    def fit(self, input_sequence, labels, epoch):
        self.model.fit(input_sequence, labels, shuffle=True, epochs=epoch, class_weight={0: 1, 1: 2, 2: 3})
        return 0

    def predict(self, input_sequence):
        """
        :param X:
        :return: outputs:类别概率分布，用于计算多类AUC
                 predictions：最终分类类别，用于计算g-mean，precision，recall，f1
        """
        score = self.model.predict(input_sequence)  # for AUC
        predictions = np.argmax(score, axis=1)  # for else
        return predictions, score


if __name__ == '__main__':
    flags = tf.compat.v1.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_integer('graph_len', 100, 'Number of sentences length.')
    flags.DEFINE_integer('features_dim', 300, 'word vector length.')
    flags.DEFINE_integer('hidden', 100, 'Number of hidden.')

    filepath = '../../Dataset/SelectedData/'
    files = os.listdir(filepath)
    save_path = r'../../Experiment Results/GGNN/'
    vocab = open(r'../MyGNNTF2/data/vocab.txt').readline().strip().split()

    # # within project
    # design, requirement = [], []
    # label_names = ['Non-SATD', 'Design', 'Requirement']
    # metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    # cur_model = GGNN(graph_len=FLAGS.graph_len, features_dim=FLAGS.features_dim,
    #                  vocab_size=len(vocab), hidden=FLAGS.hidden)
    # train_data, test_data, train_labels, test_labels = within_project(filepath, testsize=0.3)
    # cur_model.fit(train_data, train_labels, epoch=3)
    # for key in test_data.keys():
    #     x_test, y_test = test_data[key], test_labels[key]
    #
    #     y_pred, y_score = cur_model.predict(x_test)
    #     result = performence(y_test, y_pred, y_score, label_names)
    #     design.append(result[1, :].reshape(-1)), requirement.append(result[2, :].reshape(-1))
    # de_df = pd.DataFrame(design, index=files, columns=metric)
    # re_df = pd.DataFrame(requirement, index=files, columns=metric)
    # de_df.to_csv(save_path + 'WithinProject_Design.csv')
    # re_df.to_csv(save_path + 'WithinProject_Requirement.csv')

    # cross project
    # design, requirement = [], []
    # label_names = ['Non-SATD', 'Design', 'Requirement']
    # metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    # for file in files:
    #     x_train, x_test, y_train, y_test = cross_project(filepath, file)
    #     cur_model = GGNN(graph_len=FLAGS.graph_len, features_dim=FLAGS.features_dim,
    #                      vocab_size=len(vocab), hidden=FLAGS.hidden)
    #     cur_model.fit(x_train, y_train, epoch=3)
    #     y_pred, y_score = cur_model.predict(x_test)
    #     result = performence(y_test, y_pred, y_score, label_names)
    #     design.append(result[1, :].reshape(-1)), requirement.append(result[2, :].reshape(-1))
    # design, requirement = np.vstack(design), np.vstack(requirement)
    # de_df = pd.DataFrame(design, index=files, columns=metric)
    # re_df = pd.DataFrame(requirement, index=files, columns=metric)
    # de_df.to_csv(save_path+'CrossProject_Design.csv')
    # re_df.to_csv(save_path+'CrossProject_Requirement.csv')

    de, re = [], []
    label_names = ['Non-SATD', 'Design', 'Requirement']
    metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    for _ in range(20):
        x_train, x_test, y_train, y_test = mix_project(filepath, testsize=0.1)
        cur_model = GGNN(graph_len=FLAGS.graph_len, features_dim=FLAGS.features_dim,
                         vocab_size=len(vocab), hidden=FLAGS.hidden)
        cur_model.fit(x_train, y_train, epoch=3)
        y_pred, y_score = cur_model.predict(x_test)
        result = performence(y_test, y_pred, y_score, label_names)
        de.append(result[1, :]), re.append(result[-1, :])
    de = pd.DataFrame(de, columns=metric)  # 只需要 设计债和需求债
    re = pd.DataFrame(re, columns=metric)  # 只需要 设计债和需求债
    de.to_csv(save_path+'Mix_project_Design.csv')
    re.to_csv(save_path+'Mix_project_Requirement.csv')
