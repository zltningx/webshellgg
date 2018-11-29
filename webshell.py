from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics, svm
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import os
import numpy as np
import pickle


def load_file(file_path):
    t = b''
    with open(file_path, "rb") as f:
        for line in f:
            line = line.strip(b'\r\n')
            t += line
    return t


def load_files(path):
    files_list = []
    for r, d, files in os.walk(path):
        for file in files:
            if file.endswith('.php'):
                file_path = r + '/' + file
                print("Load {}".format(file_path))
                t = load_file(file_path)
                files_list.append(t)
    return files_list


def get_feature_by_bag_cv_tfidf():
    webshell_files = load_files(webshell_dir)
    wp_files = load_files(normal_dir)
    black_count = len(webshell_files)
    white_count = len(wp_files)
    y1 = [1]*black_count
    y2 = [0]*white_count

    x = webshell_files + wp_files
    y = y1 + y2

    cv = CountVectorizer(ngram_range=(2, 2), decode_error="ignore",
                         token_pattern=r'\b\w+\b', min_df=1,
                         max_df=1.0, max_features=max_features)
    x = cv.fit_transform(x).toarray()

    transformer = TfidfTransformer(smooth_idf=False)
    x = transformer.fit_transform(x).toarray()

    joblib.dump(cv, "cv.pkl")
    joblib.dump(transformer, "transformer.pkl")
    return x, y, cv, transformer


def do_metrics(y_test,y_pred):
    print("metrics.accuracy_score:")
    print(metrics.accuracy_score(y_test, y_pred))
    print("metrics.confusion_matrix:")
    print(metrics.confusion_matrix(y_test, y_pred))
    print("metrics.precision_score:")
    print(metrics.precision_score(y_test, y_pred))
    print("metrics.recall_score:")
    print(metrics.recall_score(y_test, y_pred))
    print("metrics.f1_score:")
    print(metrics.f1_score(y_test, y_pred))


def do_mlp(x, y):
    clf = MLPClassifier(solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2),
                        random_state=1)

    # train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(np.mean(y_test==clf.predict(x_test)))
    do_metrics(y_test, y_pred)

    # save
    joblib.dump(clf, "mlp.pkl", compress=3)


def do_svm(x, y):
    clf = svm.SVC()

    # train
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    do_metrics(y_test, y_pred)
    joblib.dump(clf, "svm.pkl", compress=3)


def do_gnb(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    gnb = GaussianNB()
    gnb.fit(x_train, y_train)
    y_pred = gnb.predict(x_test)
    do_metrics(y_test, y_pred)
    joblib.dump(gnb, "gnb.pkl", compress=3)


def check(clf, cv, transformer, path):
    all = 0
    webshell = 0
    not_webshell = 0
    # webshell_files_list = load_files(webshell_dir)
    for r, d, files in os.walk(path):
        for file in files:
            file_path = r + '/' + file
            t = load_file(file_path)
            t_list = list()
            t_list.append(t)
            x = cv.transform(t_list).toarray()
            x = transformer.transform(x).toarray()
            y_pred = clf.predict(x)
            all += 1
            if y_pred[0] == 1:
                print("{} is webshell".format(file_path))
                webshell += 1
            else:
                print("not")
                not_webshell += 1
    print("全部文件：{}，webshell {} 未检出：{} 检出率：{}".format(all, webshell, not_webshell, webshell/all))


max_features = 25000
webshell_dir = "webshell/php"
normal_dir = "normal"

if __name__ == '__main__':
    # train
    # x, y, cv, transformer = get_feature_by_bag_cv_tfidf()
    # do_mlp(x, y)
    check(joblib.load("mlp.pkl"), joblib.load("cv.pkl"), joblib.load("transformer.pkl"),
          "normal")
