from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GRU, Embedding
from tensorflow.keras.layers import Dot, Bidirectional
from origin_Attention import Attention
from Tools import *
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split


class MyLCM:
    def __init__(self, sen_len, vocab_size, wvdim, hidden_size, num_classes=3, alpha=0.5, text_embedding_matrix=None, label_embedding_Dense=True):
        self.num_classes = num_classes
        self.alpha = alpha
        # basic_predictor:
        text_input = Input(shape=(sen_len,), name='text_input')
        if text_embedding_matrix is None:
            input_emb = Embedding(vocab_size, wvdim, input_length=sen_len, name='text_emb', mask_zero=True)(text_input)  # (V,wvdim)
        else:
            input_emb = Embedding(vocab_size, wvdim, input_length=sen_len, weights=[text_embedding_matrix],
                                  name='text_emb')(text_input)  # (V,wvdim)
        input_vec = Bidirectional(GRU(hidden_size, return_sequences=True))(input_emb)
        attention_layer = Attention(step_dim=input_vec.shape[1])(input_vec)
        pred_probs = Dense(num_classes, activation='softmax', name='pred_probs')(attention_layer)
        self.basic_predictor = Model(inputs=text_input, outputs=pred_probs)
        self.basic_predictor.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam')

        # LCM:
        label_input = Input(shape=(num_classes,), name='label_input')
        if label_embedding_Dense == True:
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(label_input)  # (n,wvdim)
            label_emb = Dense(hidden_size*2, activation='tanh', name='label_emb2')(label_emb)
        else:
            label_emb = Embedding(num_classes, wvdim, input_length=num_classes, name='label_emb1')(label_input)  # (n,wvdim)
        # similarity part:
        doc_product = Dot(axes=(2, 1))([label_emb, attention_layer])  # (n,d) dot (d,1) --> (n,1)
        label_sim_dict = Dense(num_classes, activation='softmax', name='label_sim_dict')(doc_product)
        # concat output:
        concat_output = self.alpha*label_sim_dict + (1-self.alpha)*pred_probs
        # compile；
        self.model = Model(inputs=[text_input, label_input], outputs=concat_output)
        self.model.compile(loss=keras.losses.SparseCategoricalCrossentropy(), optimizer='adam')

    def my_evaluator(self, model, inputs):
        """
        :param model:
        :param inputs:
        :return: outputs:类别概率分布，用于计算多类AUC
                 predictions：最终分类类别，用于计算g-mean，precision，recall，f1
        """
        score = model.predict(inputs)  # for AUC
        predictions = np.argmax(score, axis=1)  # for else
        return score, predictions

    def train_val(self, data_package, epochs, class_weights, save_best=False):
        x_train, y_train, x_test, y_test = data_package
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        x_test = np.array(x_test)
        y_test = np.array(y_test)

        l_train = np.array([np.array(range(self.num_classes)) for i in range(len(x_train))])
        l_test = np.array([np.array(range(self.num_classes)) for i in range(len(x_test))])
        target_names = ['non-SATD', 'Design-SATD', 'Require-SATD']
        best_performence = 0
        best_f1 = 0
        self.model.fit([x_train, l_train], y_train, epochs=epochs, class_weight=class_weights)
        final_test_score, final_test_predictions = self.my_evaluator(self.model, [x_test, l_test])
        performence = analysis(y_true=y_test, y_pred=final_test_predictions, y_score=final_test_score)
        return np.mean(performence[2])

if __name__ == '__main__':
    sen_len, wv_len = 50, 100
    sentences = file2sentences(r'C:\Users\xlg\Desktop\论文2\SATD数据集\实验用\Comments1.txt')
    projects = open(r'C:\Users\xlg\Desktop\论文2\SATD数据集\实验用\Projects.txt').readlines()
    labels = open(r'C:\Users\xlg\Desktop\论文2\SATD数据集\实验用\Labels.txt').readlines()
    labels = [int(x) for x in labels]

    w2vmodel = Word2Vec(sentences=sentences, vector_size=wv_len, window=5, min_count=3, workers=4)
    vocab = list(w2vmodel.wv.key_to_index.keys())
    vocab.insert(0, '')
    sentences_idx = word2index(vocab, sentences, sen_len)

    random_state = 1
    x_train, x_test, y_train, y_test = train_test_split(
        sentences_idx, labels, test_size=0.1, random_state=random_state)
    data_package = [x_train, y_train, x_test, y_test]

    result = np.zeros(shape=(4, 4), dtype=np.float32)
    de, re = [1, 2, 3, 4], [1, 2, 3, 4]
    for i in range(len(de)):
        for j in range(len(re)):
            class_weights = {0: 1, 1: de[i], 2: re[j]}
            lcm_model = GRULCM(sen_len=sen_len, hidden_size=sen_len, num_classes=3, alpha=0.6, vocab_size=len(vocab),
                               wvdim=wv_len, label_embedding_Dense=True)
            F1 = lcm_model.train_val(data_package=data_package, epochs=3, class_weights=class_weights)
            result[i][j] = F1
            print(F1)
            lcm_model.model.reset_states()
    print(result)










