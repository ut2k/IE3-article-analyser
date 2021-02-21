import json
import csv
import nltk
import pandas as pd
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer

classifiers = {
    # "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    # "MultinomialNB": MultinomialNB(),
    # "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=2000),
    # "MLPClassifier": MLPClassifier(max_iter=7500),
    "AdaBoostClassifier": AdaBoostClassifier(),
}


def main():
    print("\nCleaning json...")
    clean_json('../released_data.json', '../Dataset/cleaned_data.json')
    # clean_json('../Dataset/test_data.json', '../Dataset/cleaned_data.json')

    print("\nReading json...")
    # reads json data into a pandas dataframe object
    data = pd.read_json('../Dataset/cleaned_data.json')
    data.info()

    print("\nCV training...")
    # count_vectorization(data)
    count_vectorization(data, "AdaBoost", '../Dataset/test_article.json')

    print("\nTF training...")
    tf_idf(data, "AdaBoost", '../Dataset/test_article.json')

    return


# data: full dataset to be trained and tested on
# target_clf (optional): the name of the specific classifier to be used
# target_path (optional): file path for new untagged article (must be json)
def count_vectorization(data: pd.DataFrame, target_clf="", target_path=""):
    # generates document term matrix using CV
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # removes stop words, symbols and change to lowercase
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    # learn(fit) vocab dictionary and return doc-term matrix
    text_counts = cv.fit_transform(data['content'])

    # split into training and testing sets with random state
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['allsides_bias'], test_size=0.3, random_state=123)

    # trains model and compares with testing set
    # name: name of classifier, sklearn_clf: actual classifier
    def fit_and_predict(name, sklearn_clf):
        clf = sklearn_clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predicted)
        print(F"{accuracy:.2%} - {name}")

        # if target_clf provided, will run trained model on test case at target_path
        if len(target_clf) > 0 and len(target_path):
            article = pd.read_json(target_path)
            # transform new article into doc-term matrix
            article_counts = cv.transform(article['content'])
            # run trained model on new article
            predicted = clf.predict(article_counts)
            print("Reading new articles:...")
            for i in range(len(predicted)):
                print("\"" + article['title'][i][:20] + "...\" : " + predicted[i])

    # trains and predicts for the target_clf only
    for name, sklearn_clf in classifiers.items():
        if len(target_clf) > 0 and target_clf in name:
            fit_and_predict(name, sklearn_clf)
            return

    # trains and predicts for all classifiers
    for name, sklearn_clf in classifiers.items():
        fit_and_predict(name, sklearn_clf)

    return


# use tf-idf to normalize document term matrix
def tf_idf(data: pd.DataFrame, target_clf="", target_path=""):
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    tf = TfidfVectorizer(lowercase=True, stop_words='english', tokenizer=token.tokenize)
    text_tf = tf.fit_transform(data['content'])
    X_train, X_test, y_train, y_test = train_test_split(
        text_tf, data['allsides_bias'], test_size=0.3, random_state=123)

    # trains model and compares with testing set
    # name: name of classifier, sklearn_clf: actual classifier
    def fit_and_predict(name, sklearn_clf):
        clf = sklearn_clf.fit(X_train, y_train)
        predicted = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predicted)
        print(F"{accuracy:.2%} - {name}")

        # if target_clf provided, will run trained model on test case at target_path
        if len(target_clf) > 0 and len(target_path):
            article = pd.read_json(target_path)
            # transform new article into doc-term matrix
            article_counts = tf.transform(article['content'])
            # run trained model on new article
            predicted = clf.predict(article_counts)
            print("Reading new articles:...")
            for i in range(len(predicted)):
                print("\"" + article['title'][i][:20] + "...\" : " + predicted[i])

    # trains and predicts for the target_clf only
    for name, sklearn_clf in classifiers.items():
        if len(target_clf) > 0 and target_clf in name:
            fit_and_predict(name, sklearn_clf)
            return

    # trains and predicts for all classifiers
    for name, sklearn_clf in classifiers.items():
        fit_and_predict(name, sklearn_clf)

    return


def clean_json(path_r: str, path_w: str):
    with open(path_r, 'r') as f_r:
        with open(path_w, 'w') as f_w:
            count = 0
            for line in f_r.readlines():
                if count == 0:
                    f_w.write("[\n")
                    count += 1
                    continue
                if count == 1:
                    f_w.write(line)
                    count += 1
                    continue
                f_w.write("," + line)

            f_w.write("\n]")


if __name__ == '__main__':
    main()
