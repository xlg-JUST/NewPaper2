import numpy as np
from sklearn.ensemble import RandomForestClassifier
from Tools import *
import os

# a = [0, 1, 2, 3, 4]
# temp = []
# ngram = []
# for word in a:
#     temp.append(str(word))
# for tmp in zip(*[temp[i:] for i in range(2)]):


class NgramIDF:
    def __init__(self):
        self.clf = RandomForestClassifier()
        self.vocab = set([])
        self.features = []

    def features_selection(self, x_train, y_train):
        """
        only used when first build model
        calculate gram's tfidf and return high-tfidf grams
        :param x_train:
        :param y_train:
        :return:
        """
        for grams in x_train:
            self.vocab = set(self.vocab | set(grams))
        self.vocab = list(self.vocab)
        volen = len(self.vocab)
        de_tf, re_tf, idf = [0]*volen, [0]*volen, [0]*volen
        de_wordcount, re_wordcount = 0, 0
        total = len(x_train)
        for i in range(total):
            tmp = list(set(x_train[i]))
            for gram in tmp:
                idf[self.vocab.index(gram)] += 1
            if y_train[i] == 0:
                continue
            elif y_train[i] == 1:
                de_wordcount += len(x_train[i])
                for gram in x_train[i]:
                    de_tf[self.vocab.index(gram)] += 1
            elif y_train[i] == 2:
                re_wordcount += len(x_train[i])
                for gram in x_train[i]:
                    re_tf[self.vocab.index(gram)] += 1
        de_tf = [tf/de_wordcount for tf in de_tf]
        re_tf = [tf/re_wordcount for tf in re_tf]
        idf = [np.log(total/(IDF+1)) for IDF in idf]
        de_tfidf = [a*b for a, b in zip(de_tf, idf)]
        re_tfidf = [a*b for a, b in zip(re_tf, idf)]
        tfidf = [np.sqrt(a*b) for a, b in zip(de_tfidf, re_tfidf)]
        tfdict = {a: b for a, b in zip(self.vocab, tfidf)}
        tfdict = sorted(tfdict.items(), key=lambda d: d[1], reverse=True)
        features = [tup[0] for tup in tfdict][:int(len(tfdict)/10)]  # 10% features
        features.insert(0, '[PAD]')
        self.features = features
        return 0

    def save_features(self):
        file = open(r'../NgramIDF/features.txt', 'w', encoding='utf-8')
        for gram in self.features:
            file.write(gram+' ')
        file.close()
        return 0

    def load_features(self, path, fn=2400):
        features = open(path, encoding='utf-8').readline().strip().split()[1:fn+1]
        self.features = features
        return features

    def fit(self, x_train, y_train):
        self.clf.fit(x_train, y_train)
        return 0

    def predict(self, x_test):
        y_pred = self.clf.predict(x_test)
        y_score = self.clf.predict_proba(x_test)
        return y_pred, y_score


if __name__ == '__main__':
    """
    ????????????
    """
    vocab = open(r'../../Dataset/Vocab/min_count=5.txt', encoding='utf-8').readline().strip().split()

    filepath = '../../Dataset/SelectedData/'
    save_path = r'../../Experiment Results/NgramIDF/'
    files = os.listdir(filepath)

    # ??????gram
    # x_train, _, y_train, _ = mix_project(filepath, testsize=0.1)
    # x_train = str2list(x_train)
    # x_train = word2index(x_train, vocab)
    # x_train = index2gram(x_train)
    # clf = NgramIDF()
    # clf.features_selection(x_train, y_train)
    # clf.save_features()



    # design, requirement = [], []
    # label_names = ['Non-SATD', 'Design', 'Requirement']
    # metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    # for file in files:
    #     x_train, x_test, y_train, y_test = cross_project(filepath, file)
    #     x_train, x_test = str2list(x_train), str2list(x_test)
    #     x_train, x_test = word2index(x_train, vocab), word2index(x_test, vocab)
    #     x_train, x_test = index2gram(x_train), index2gram(x_test)
    #     cur_model = NgramIDF()
    #     gram_vocab = cur_model.load_features(r'../NgramIDF/features.txt', fn=600)
    #     x_train, x_test = build_bow(x_train, gram_vocab), build_bow(x_test, gram_vocab)
    #     cur_model.fit(x_train, y_train)
    #     y_pred, y_score = cur_model.predict(x_test)
    #     result = performence(y_test, y_pred, y_score, label_names)
    #     design.append(result[1, :].reshape(-1)), requirement.append(result[2, :].reshape(-1))
    #     print(file+'  Done!')
    # design, requirement = np.vstack(design), np.vstack(requirement)
    # de_df = pd.DataFrame(design, index=files, columns=metric)
    # re_df = pd.DataFrame(requirement, index=files, columns=metric)
    # de_df.to_csv(save_path+'CrossProject_Design.csv')
    # re_df.to_csv(save_path+'CrossProject_Requirement.csv')

    design, requirement = [], []
    label_names = ['Non-SATD', 'Design', 'Requirement']
    metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    train_data, train_labels, test_data, test_labels = within_project(filepath, testsize=0.3)
    cur_model = NgramIDF()
    gram_vocab = cur_model.load_features(r'../NgramIDF/features.txt')
    train_data = build_bow(index2gram(word2index(str2list(train_data), vocab)), gram_vocab)
    cur_model.fit(train_data, train_labels)
    for key in test_data.keys():
        x_test, y_test = test_data[key], test_labels[key]
        x_test = build_bow(index2gram(word2index(str2list(x_test), vocab)), gram_vocab)
        y_pred, y_score = cur_model.predict(x_test)
        result = performence(y_test, y_pred, y_score, label_names)
        design.append(result[1, :].reshape(-1)), requirement.append(result[2, :].reshape(-1))
    de_df = pd.DataFrame(design, index=files, columns=metric)
    re_df = pd.DataFrame(requirement, index=files, columns=metric)
    de_df.to_csv(save_path + 'WithinProject_Design.csv')
    re_df.to_csv(save_path + 'WithinProject_Requirement.csv')



