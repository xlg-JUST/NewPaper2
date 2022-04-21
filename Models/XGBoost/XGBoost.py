import xgboost as xgb
import os
from Tools import *


class XGboost:
    def __init__(self):
        self.clf = xgb.XGBClassifier()
        self.class_weights = [1, 1, 1]

    def fit(self, x_train, y_train):
        sample_weights = [self.class_weights[label] for label in y_train]
        self.clf.fit(x_train, y_train, sample_weight=sample_weights)
        return 0

    def predict(self, x_test):
        y_pred = self.clf.predict(x_test)
        y_score = self.clf.predict_proba(x_test)
        return y_pred, y_score


if __name__ == '__main__':
    """
    准备数据
    """
    vocab = open(r'../../Dataset/Vocab/min_count=5.txt', encoding='utf-8').readline().strip().split()

    filepath = '../../Dataset/SelectedData/'
    save_path = r'../../Experiment Results/XGBoost/'
    files = os.listdir(filepath)

    # design, requirement = [], []
    # label_names = ['Non-SATD', 'Design', 'Requirement']
    # metric = ['Precision', 'Recall', 'F1', 'Gmean', 'AUC']
    # for file in files:
    #     x_train, x_test, y_train, y_test = cross_project(filepath, file)
    #     x_train, x_test = build_bow(x_train, vocab, iflist=False), build_bow(x_test, vocab, iflist=False)
    #     cur_model = XGboost()
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
    cur_model = XGboost()
    train_data = build_bow(train_data, vocab, iflist=False)
    cur_model.fit(train_data, train_labels)
    for key in test_data.keys():
        x_test, y_test = test_data[key], test_labels[key]
        x_test = build_bow(x_test, vocab, iflist=False)
        y_pred, y_score = cur_model.predict(x_test)
        result = performence(y_test, y_pred, y_score, label_names)
        design.append(result[1, :].reshape(-1)), requirement.append(result[2, :].reshape(-1))
    de_df = pd.DataFrame(design, index=files, columns=metric)
    re_df = pd.DataFrame(requirement, index=files, columns=metric)
    de_df.to_csv(save_path + 'WithinProject_Design.csv')
    re_df.to_csv(save_path + 'WithinProject_Requirement.csv')
