from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
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
            # if file.endswith('.php'):
                file_path = r + '/' +file
                print ("Load {}".format(file_path))
                t = load_file(file_path)
                files_list.append(t)
    return files_list


def pre_date(x):

    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                          token_pattern = r'\b\w+\b',min_df=1)
    x = cv.fit_transform(x).toarray()

    transformer = TfidfTransformer(smooth_idf=False)

    return transformer.fit_transform(x).toarray()


def check():

    webshell_files_list=load_files("./webshell/php/PHP-WEBSHELL")
    wp_files_list = load_files("./normal/")
    test = load_file("2.php")

    x1 = pre_date(test)
    x2 = pre_date(wp_files_list)

    # x = np.concatenate((x1, x2))
    y = np.concatenate(([1]*len(webshell_files_list), [0]*len(wp_files_list)))

    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    # clf = GaussianNB()
    # clf.fit(x_train, y_train)
    # joblib.dump(clf, "clf.pkl")

    clf = joblib.load("clf.pkl")

    print(clf.predict(x1[-1:]))


if __name__ == '__main__':
    check()