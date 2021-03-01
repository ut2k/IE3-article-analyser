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
import itertools

import csv

tqdm.pandas(desc="progress-bar")

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

    print("\nIBC training...")
    ibc_classify(data)

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


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text

    # TODO try to pickle the matrices + display truncated data structure


# use pipeline to normalize document term matrix
def ibc_classify(data: pd.DataFrame):
    data['content'].apply(clean_text)
    tokens = RegexpTokenizer(r'[a-zA-Z]+')
    cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words='english', ngram_range=(1, 3))

    # row: document number, col: feature frequency, ordered by get_features_names()
    text_counts = cv.fit_transform(data['content'])
    feature_list = cv.get_feature_names()
    print(text_counts.shape)

    # turn feature_list into a dict with the index as value -> random access
    feature_dict = {feature_list[i]: i for i in range(0, len(feature_list))}

    NEU_LEN = 14846
    LIB_LEN = 4448
    CON_LEN = 4448
    ROW_LEN = text_counts.shape[0]

    with open("./../Dataset/ibc_data/feature_lists/neu_list.csv", 'r') as f:
        reader = csv.DictReader(f)
        pbar = tqdm(total=NEU_LEN)
        for row in reader:
            if row['gram'] == '1':
                if row['1st'] in feature_dict:
                    i = feature_dict[row['1st']]
                    for doc_i in range(ROW_LEN):
                        if text_counts[doc_i, i] > 0:
                            text_counts[doc_i, i] += 1.0 * float(row['freq']) / 10
            if row['gram'] == '2':
                word = F"{row['1st']} {row['2nd']}"
                if word in feature_dict:
                    i = feature_dict[word]
                    for doc_i in range(ROW_LEN):
                        if text_counts[doc_i, i] > 0:
                            text_counts[doc_i, i] += 1.0 * float(row['freq']) / 2
            if row['gram'] == '3':
                word = F"{row['1st']} {row['2nd']} {row['3rd']}"
                if word in feature_dict:
                    i = feature_dict[word]
                    for doc_i in range(ROW_LEN):
                        if text_counts[doc_i, i] > 0:
                            text_counts[doc_i, i] += 1.0 * float(row['freq']) / 1.5
            pbar.update(1)
        pbar.close()

    with open("./../Dataset/ibc_data/feature_lists/lib_list.csv", 'r') as f:
        reader = csv.DictReader(f)
        pbar = tqdm(total=LIB_LEN)
        for row in reader:
            if row['gram'] == '1':
                if row['1st'] in feature_dict:
                    i = feature_dict[row['1st']]
                    for doc_i in range(ROW_LEN):
                        if (data['allsides_bias'][doc_i] == "From the Left"
                                and text_counts[doc_i, i] > 0):
                            text_counts[doc_i, i] += 1.0 * float(row['freq']) / 10
            if row['gram'] == '2':
                word = F"{row['1st']} {row['2nd']}"
                if word in feature_dict:
                    i = feature_dict[word]
                    for doc_i in range(ROW_LEN):
                        if (data['allsides_bias'][doc_i] == "From the Left"
                                and text_counts[doc_i, i] > 0):
                            text_counts[doc_i, i] += 1.0 * float(row['freq']) / 2
            if row['gram'] == '3':
                word = F"{row['1st']} {row['2nd']} {row['3rd']}"
                if word in feature_dict:
                    i = feature_dict[word]
                    for doc_i in range(ROW_LEN):
                        if (data['allsides_bias'][doc_i] == "From the Left"
                                and text_counts[doc_i, i] > 0):
                            text_counts[doc_i, i] += 1.0 * float(row['freq']) / 1.5

            pbar.update(1)
        pbar.close()

    with open("./../Dataset/ibc_data/feature_lists/con_list.csv", 'r') as f:
        reader = csv.DictReader(f)
        pbar = tqdm(total=CON_LEN)
        for row in reader:
            if row['gram'] == '1':
                if row['1st'] in feature_dict:
                    i = feature_dict[row['1st']]
                    for doc_i in range(ROW_LEN):
                        if (data['allsides_bias'][doc_i] == "From the Right"
                                and text_counts[doc_i, i] > 0):
                            text_counts[doc_i, i] += 1.0 * float(row['freq']) / 10
            if row['gram'] == '2':
                word = F"{row['1st']} {row['2nd']}"
                if word in feature_dict:
                    i = feature_dict[word]
                    for doc_i in range(ROW_LEN):
                        if (data['allsides_bias'][doc_i] == "From the Right"
                                and text_counts[doc_i, i] > 0):
                            text_counts[doc_i, i] += 1.0 * float(row['freq']) / 2
            if row['gram'] == '3':
                word = F"{row['1st']} {row['2nd']} {row['3rd']}"
                if word in feature_dict:
                    i = feature_dict[word]
                    for doc_i in range(ROW_LEN):
                        if (data['allsides_bias'][doc_i] == "From the Right"
                                and text_counts[doc_i, i] > 0):
                            text_counts[doc_i, i] += 1.0 * float(row['freq']) / 1.5
            pbar.update(1)
        pbar.close()

    tfidf_counts = TfidfTransformer().fit_transform(text_counts)
    # tfidf_counts = text_counts
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_counts, data['allsides_bias'], test_size=0.3, random_state=298)

    # clf = SGDClassifier().fit(X_train, y_train)
    clf = AdaBoostClassifier().fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    log_new()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    name = "AdaBoostClassifier"
    print(F"{accuracy:.2%} - ibc{name}")
    log_results(F"{accuracy:.2%} - ibc{name}")
    my_tags = ['From the Right', 'From the Left', 'From the Center']
    print(classification_report(y_test, y_pred, target_names=my_tags))

    filename = 'Models/ibc_' + name + '.sav'
    pickle.dump(clf, open(filename, 'wb'))

    return


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
