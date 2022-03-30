import pandas as pd
import re
import string
from nltk.stem.porter import PorterStemmer
import os
from gensim.models import Word2Vec
from Tools import build_vocab, str2list


def preprocess(texts):
    result = []
    stem = PorterStemmer()
    stopwords = ['is', 'was', 'are', 'were', 'for', 'the']
    for x in texts:
        x.encode('ascii', 'ignore').decode()
        x = ' '.join([word.lower() for word in x.strip().split(' ') if word not in stopwords and 2 < len(word) < 20])
        x = re.sub("https*\S+", " ", x)
        x = re.sub("www.*\S+", " ", x)
        x = re.sub("\'\w+", '', x)
        x = re.sub('[%s]' % re.escape(string.punctuation), ' ', x)
        x = re.sub(r'\d+', '', x)
        x = [stem.stem(word) for word in x.strip().split()]
        x = [word for word in x if len(word) > 1]
        result.append(' '.join(x))
    return result


def select(file):
    labels = ['WITHOUT_CLASSIFICATION', 'DESIGN', 'IMPLEMENTATION']
    cur_df = pd.read_csv(file)
    cur_df = cur_df[cur_df['classification'].isin(labels)]
    cur_df['classification'] = cur_df['classification'].map({'WITHOUT_CLASSIFICATION': 0, 'DESIGN': 1, 'IMPLEMENTATION': 2})
    return cur_df


if __name__ == '__main__':

    source_dir = r'Dataset/SourceData/'
    preprocess_dir = r'Dataset/PreprocessData/'
    selected_dir = r'Dataset/SelectedData/'

    """
    数据预处理
    files = os.listdir(source_dir)
    for file in files:
        if file == '.DS_Store':
            continue
        df = pd.read_csv(source_dir+file)
        project_name = df['projectname'].tolist()
        classification = df['classification'].tolist()
        comments = df['commenttext'].tolist()
        sentences = preprocess(comments)
        data = {'projectname': project_name, 'classification': classification, 'preprocess_comments': sentences}
        new_df = pd.DataFrame(data)
        new_df.to_csv(preprocess_dir + file, index=False)
    """

    """
    
    属于筛选，只选择design、requirement、NonSATD 并将标签替换为1，2，0
    
    files = os.listdir(preprocess_dir)
    for file in files:
        if file == '.DS_Store':  # mac自带的文件
            continue
        df = select(preprocess_dir+file)
        df.to_csv(selected_dir+file, index=None)
    """

    """
    建立词库，分别建立低频词不同阈值的词库
    
    files = os.listdir(selected_dir)
    comments = []
    for file in files:
        if file == '.DS_Store':  # mac自带的文件
            continue
        df = pd.read_csv(selected_dir+file)
        sen = df['preprocess_comments'].to_list()
        comments.extend(sen)
    sentences = str2list(comments)
    for i in range(5):
        min_count = i+1
        vocab = build_vocab(sentences, min_count)
        f = open(r'Dataset/Vocab/min_count={}.txt'.format(min_count), 'w', encoding='utf-8')
        for word in vocab:
            f.write(word+' ')
        f.close()
    """






