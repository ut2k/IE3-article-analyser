import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from datetime import datetime
import re
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

tqdm.pandas(desc="progress-bar")
from gensim.models import Doc2Vec
from sklearn import utils
from gensim.models.doc2vec import TaggedDocument
import numpy as np

from sklearn.naive_bayes import (
    BernoulliNB,
    ComplementNB,
    MultinomialNB,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import (
    LogisticRegression,
    SGDClassifier,
)
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
    "LogisticRegression": LogisticRegression(max_iter=10000),
    "AdaBoostClassifier": AdaBoostClassifier(),
    # "MLPClassifier": MLPClassifier(max_iter=1000),
}


def main():
    print("\nCleaning json...")
    clean_json('../released_data.json', '../Dataset/cleaned_data.json')  # uncomment to ues full dataset
    # clean_json('../Dataset/test_data.json', '../Dataset/cleaned_data.json') # comment to use smaller dataset

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
    tf_idf(data) # uncomment to train all classifiers
    # tf_idf(data, clf_name)  # uncomment to train specific classifier

    print("\nPipeline training...")
    # pipeline(data)  # uncomment to train all classifiers
    # pipeline(data, clf_name)  # uncomment to train specific classifier

    print("\nDoc2Vec training...")
    # doc2Vec(data)

    print("\nOut-of-Sample Testing...")
    for name in classifiers.keys():
        if clf_name in name:
            print("CV " + name + ": ")
            cv = get_cv()
            cv.fit_transform(data['content'])
            classify_out_of_sample(name, path, cv, "cv")  # uncomment to test out of sample

            print("TF " + name + ": ")
            tf = get_tf()
            tf.fit_transform(data['content'])
            classify_out_of_sample(name, path, tf, "tf")  # uncomment to test out of sample

    return


def classify_out_of_sample(name, path, vectorizer, vec_name):
    article = pd.read_json(path)
    # transform new article into doc-term matrix
    article_counts = vectorizer.transform(article['content'])
    # run trained model on new article
    filename = "Models/" + vec_name + "_" + name + ".sav"
    loaded_clf = pickle.load(open(filename, 'rb'))
    # predicted = loaded_clf.predict(article_counts) # predict class
    predicted = loaded_clf.predict_proba(article_counts)  # predict probabilities

    for i in range(len(predicted)):
        # print("\t\"" + article['title'][i][:20] + "...\" : " + predicted[i]) # print class
        proba_str = F"L: {predicted[i][1]:.2%} | R: {predicted[i][2]:.2%} | C: {predicted[i][0]:.2%}"
        print(F"\t\"{article['title'][i][:10]}...\" | " + proba_str)  # print probabilities

    return


def get_cv():
    print("Building vectorizer...")

    # generates document term matrix using CV

    def tokenize(text):
        tokens = RegexpTokenizer(r'[a-zA-Z0-9]+').tokenize(text)
        stop_words = set(stopwords.words("english"))
        stopped_tokens = []
        for w in tokens:
            if w not in stop_words:
                stopped_tokens.append(w)
        stemmer = PorterStemmer()
        stems = stopped_tokens
        # stems = [stemmer.stem(item) for item in stopped_tokens]
        return stems

    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # removes stop words, symbols and change to lowercase
    cv = CountVectorizer(lowercase=True, stop_words=None,
                         ngram_range=(1, 1), tokenizer=tokenize,
                         min_df=0.00, strip_accents='unicode')

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
        text_counts, data['allsides_bias'], test_size=0.3, random_state=298)
    log_new()

    # trains model and compares with testing set
    # name: name of classifier, sklearn_clf: actual classifier
    def fit_and_predict(name, sklearn_clf):
        print("train...")
        clf = sklearn_clf.fit(X_train, y_train)

        # filename = "Models/cv_" + name + ".sav"
        # clf = pickle.load(open(filename, 'rb'))

        predicted = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predicted)
        print(F"{accuracy:.2%} - {name}")
        log_results(F"{accuracy:.2%} - cv{name}")
        my_tags = ['From the Right', 'From the Left', 'From the Center']
        print(classification_report(y_test, predicted, target_names=my_tags))

        # saves model by name
        filename = 'Models/cv_' + name + '.sav'
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
    print("Building vectorizer...")

    def tokenize(text):
        tokens = RegexpTokenizer(r'[a-zA-Z0-9]+').tokenize(text)
        stop_words = set(stopwords.words("english"))
        stopped_tokens = []
        for w in tokens:
            if w not in stop_words:
                stopped_tokens.append(w)
        stemmer = PorterStemmer()
        stems = stopped_tokens
        # stems = [stemmer.stem(item) for item in stopped_tokens]
        return stems

    # token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    tf = TfidfVectorizer(lowercase=True, stop_words=None,
                         tokenizer=tokenize, min_df=0.00,
                         ngram_range=(1, 1), strip_accents='unicode')

    return tf


# use tf-idf to normalize document term matrix
def tf_idf(data: pd.DataFrame, target_clf=""):
    tf = get_tf()
    text_tf = tf.fit_transform(data['content'])
    X_train, X_test, y_train, y_test = train_test_split(
        text_tf, data['allsides_bias'], test_size=0.3, random_state=123)
    log_new()

    # trains model and compares with testing set
    # name: name of classifier, sklearn_clf: actual classifier
    def fit_and_predict(name, sklearn_clf):
        clf = sklearn_clf.fit(X_train, y_train)

        # filename = "Models/tf_" + name + ".sav"
        # clf = pickle.load(open(filename, 'rb'))

        predicted = clf.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, predicted)
        print(F"{accuracy:.2%} - {name}")
        log_results(F"{accuracy:.2%} - tf{name}")
        my_tags = ['From the Right', 'From the Left', 'From the Center']
        print(classification_report(y_test, predicted, target_names=my_tags))

        # saves model by name
        filename = 'Models/tf_' + name + '.sav'
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


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text


# use pipeline to normalize document term matrix
def pipeline(data: pd.DataFrame):
    data['content'].apply(clean_text)
    X_train, X_test, y_train, y_test = train_test_split(
        data['content'], data['allsides_bias'], test_size=0.3, random_state=42)

    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', SGDClassifier()),
                   ])

    nb.fit(X_train, y_train)
    predicted = nb.predict(X_test)
    log_new()
    accuracy = metrics.accuracy_score(y_test, predicted)
    name = "SGDClassifier"
    print(F"{accuracy:.2%} - pl{name}")
    log_results(F"{accuracy:.2%} - pl{name}")
    my_tags = ['From the Right', 'From the Left', 'From the Center']
    print(classification_report(y_test, predicted, target_names=my_tags))

    filename = 'Models/pl_' + name + '.sav'
    pickle.dump(nb, open(filename, 'wb'))

    return


def doc2Vec(data: pd.DataFrame):
    data['content'].apply(clean_text)

    def label_sentences(corpus, label_type):
        labeled = []
        for i, v in enumerate(corpus):
            label = label_type + '_' + str(i)
            labeled.append(TaggedDocument(v.split(), [label]))
        return labeled

    X_train, X_test, y_train, y_test = train_test_split(data['content'], data['allsides_bias'], random_state=13,
                                                        test_size=0.3)
    X_train = label_sentences(X_train, 'Train')
    X_test = label_sentences(X_test, 'Test')
    all_data = X_train + X_test

    model_dbow = Doc2Vec(dm=0, vector_size=300, negative=5, min_count=1, alpha=0.065, min_alpha=0.065)
    model_dbow.build_vocab([x for x in tqdm(all_data)])

    for epoch in range(30):
        model_dbow.train(utils.shuffle([x for x in tqdm(all_data)]), total_examples=len(all_data), epochs=1)
        model_dbow.alpha -= 0.002
        model_dbow.min_alpha = model_dbow.alpha

    def get_vectors(model, corpus_size, vectors_size, vectors_type):
        vectors = np.zeros((corpus_size, vectors_size))
        for i in range(0, corpus_size):
            prefix = vectors_type + '_' + str(i)
            vectors[i] = model.docvecs[prefix]
        return vectors

    train_vectors_dbow = get_vectors(model_dbow, len(X_train), 300, 'Train')
    test_vectors_dbow = get_vectors(model_dbow, len(X_test), 300, 'Test')

    logreg = LogisticRegression(n_jobs=1, C=1e5, max_iter=10000)
    logreg.fit(train_vectors_dbow, y_train)
    logreg = logreg.fit(train_vectors_dbow, y_train)
    y_pred = logreg.predict(test_vectors_dbow)
    log_new()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    name = "LogisticRegression"
    print(F"{accuracy:.2%} - d2v{name}")
    log_results(F"{accuracy:.2%} - d2v{name}")
    my_tags = ['From the Right', 'From the Left', 'From the Center']
    print(classification_report(y_test, y_pred, target_names=my_tags))

    filename = 'Models/d2v_' + name + '.sav'
    pickle.dump(logreg, open(filename, 'wb'))


def log_results(result: str):
    with open("results.txt", 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%H:%M:%S %d/%m/%Y");
        f.write(dt_string + " - " + result + "\n")


def log_new():
    with open("results.txt", 'a') as f:
        now = datetime.now()
        dt_string = now.strftime("%H:%M:%S %d/%m/%Y");
        f.write("\n" + dt_string + " - New training session:\n")


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
