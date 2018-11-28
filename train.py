from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.externals import joblib

import numpy as np
import os


def load_file(file_path):
    t=b""
    with open(file_path, 'rb') as f:
        for line in f:
            line=line.strip(b'\n')
            t+=line
    return t


def load_files(path):
    files_list=[]
    for r, d, files in os.walk(path):
        for file in files:
            if file.endswith('.php'):
                file_path = r + '/' +file
                print ("Load {}".format(file_path))
                t = load_file(file_path)
                files_list.append(t)
    return  files_list


def train():
    # train black list
    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                                                 token_pattern = r'\b\w+\b',min_df=1)
    webshell_files_list=load_files("./webshell/php/PHP-WEBSHELL")
    wp_files_list = load_files("./normal/")

    y_white = [0]*len(wp_files_list)
    y_black = [1]*len(webshell_files_list)

    x = wp_files_list + webshell_files_list
    y = y_white + y_black

    x = cv.fit_transform(x).toarray()

    transformer = TfidfTransformer(smooth_idf=False)

    x = transformer.fit_transform(x).toarray()
    # # train white list
    # wp_bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
    #                                        token_pattern=r'\b\w+\b', min_df=1, vocabulary=vocabulary)
    # wp_files_list = load_files("./normal/")
    # x2 = wp_bigram_vectorizer.fit_transform(wp_files_list).toarray()
    # y2 = [0] * len(x2)
    #
    # x = np.concatenate((x1, x2))
    # y = np.concatenate((y1, y2))

    # #test
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    print(x_test)
    clf = GaussianNB()
    clf.fit(x_train, y_train)
    print(clf.predict([x[0]]))

    joblib.dump(clf, './clf.pkl')

    print(np.mean(y_test==clf.predict(x_test)))



if __name__ == '__main__':
    train()
