import pickle
import pandas as pd
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

# comment/uncomment to choose classifiers to be trained
classifiers = {
    # "BernoulliNB": BernoulliNB(),
    "ComplementNB": ComplementNB(),
    # "MultinomialNB": MultinomialNB(),
    # "KNeighborsClassifier": KNeighborsClassifier(),
    "DecisionTreeClassifier": DecisionTreeClassifier(),
    "RandomForestClassifier": RandomForestClassifier(),
    "LogisticRegression": LogisticRegression(max_iter=2000),
    "MLPClassifier": MLPClassifier(max_iter=500),
    "AdaBoostClassifier": AdaBoostClassifier(),
}


def main():
    print("\nCleaning json...")
    clean_json('../released_data.json', '../Dataset/cleaned_data.json') # uncomment to ues full dataset
    # clean_json('../Dataset/small_test_data.json', '../Dataset/cleaned_data.json') # comment to use smaller dataset

    print("\nReading json...")
    # reads json data into a pandas dataframe object
    data = pd.read_json('../Dataset/cleaned_data.json')
    data.info()

    clf_name = "AdaBoostClassifier"
    path = '../Dataset/test_article.json'

    print("\nCV training...")
    # count_vectorization(data)  # uncomment to train all classifiers
    count_vectorization(data, clf_name) # uncomment to train specific classifier

    print("\nTF training...")
    # tf_idf(data) # uncomment to train all classifiers
    tf_idf(data, clf_name) # uncomment to train specific classifier

    print("\nCV Test Out of Sample...")
    cv = get_cv()
    cv.fit_transform(data['content'])
    classify_out_of_sample(clf_name, path, cv, "cv")  # uncomment to test out of sample

    print("\nTF Test Out of Sample...")
    tf = get_tf()
    tf.fit_transform(data['content'])
    classify_out_of_sample(clf_name, path, tf, "tf") # uncomment to test out of sample

    return


def classify_out_of_sample(clf_name, path, vectorizer, vec_name):
    article = pd.read_json(path)
    # transform new article into doc-term matrix
    article_counts = vectorizer.transform(article['content'])
    # run trained model on new article
    filename = "Models/" + clf_name + "_"+ vec_name +"model.sav"
    loaded_clf = pickle.load(open(filename, 'rb'))
    predicted = loaded_clf.predict(article_counts)

    for i in range(len(predicted)):
        print("\"" + article['title'][i][:20] + "...\" : " + predicted[i])

    return


def get_cv():
    # generates document term matrix using CV
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # removes stop words, symbols and change to lowercase
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)

    return cv


# data: full dataset to be trained and tested on
# target_clf (optional): the name of the specific classifier to be used
# target_path (optional): file path for new untagged article (must be json)
def count_vectorization(data: pd.DataFrame, target_clf=""):
    cv = get_cv()
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

        # saves model by name
        filename = 'Models/' + name + '_cvmodel.sav'
        pickle.dump(clf, open(filename, 'wb'))

    # trains and predicts for the target_clf only
    for name_, sklearn_clf_ in classifiers.items():
        if len(target_clf) > 0 and target_clf in name_:
            fit_and_predict(name_, sklearn_clf_)
            return

    # trains and predicts for all classifiers
    for name_, sklearn_clf_ in classifiers.items():
        fit_and_predict(name_, sklearn_clf_)

    return


def get_tf():
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    tf = TfidfVectorizer(lowercase=True, stop_words='english', tokenizer=token.tokenize)

    return tf


# use tf-idf to normalize document term matrix
def tf_idf(data: pd.DataFrame, target_clf=""):
    tf = get_tf()
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

        # saves model by name
        filename = 'Models/' + name + '_tfmodel.sav'
        pickle.dump(clf, open(filename, 'wb'))

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
