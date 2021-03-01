import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from datetime import datetime
import re
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from tqdm import tqdm

import csv

tqdm.pandas(desc="progress-bar")

from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier



def main():
    print("\nCleaning json...")
    clean_json('../released_data.json', '../Dataset/cleaned_data.json')  # uncomment to ues full dataset
    # clean_json('../Dataset/test_data.json', '../Dataset/cleaned_data.json') # comment to use smaller dataset

    print("\nReading json...")
    # reads json data into a pandas dataframe object
    data = pd.read_json('../Dataset/cleaned_data.json')
    data.info()

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


# use pipeline to normalize document term matrix
def ibc_classify(data: pd.DataFrame):
    print("First 5 docs:")
    for i in range(5):
        print(F"{i + 1}: {data['content'][i][:50]}...")

    # print("\nClean text...")
    # data['content'] = data['content'].apply(clean_text)
    # print("First 5 docs:")
    # for i in range(5):
    #     print(F"{i + 1}: {data['content'][i][:50]}...")

    tokens = RegexpTokenizer(r'[a-zA-Z]+')
    cv = CountVectorizer(tokenizer=tokens.tokenize, stop_words="english", ngram_range=(1, 3))

    # row: document number, col: feature frequency, ordered by get_features_names()
    print("\nGenerate bag of words matrix...")
    text_counts = cv.fit_transform(tqdm(data['content']))  # returns a sparse matrix, entry = matrix[x, y]
    M_SIZE = text_counts.shape
    DISPLAY_INDEX = 2901508  # index for the word "people"
    print(F"Matrix size: {M_SIZE}")

    feature_list = cv.get_feature_names()
    print(F"Ex. features: {feature_list[DISPLAY_INDEX : DISPLAY_INDEX + 5]}")
    print(F"Ex. features in first 5 docs:")
    for i in range(5):
        for j in range(DISPLAY_INDEX, DISPLAY_INDEX + 5):
            print(F"\t{text_counts[i, j]}", end=" ")
        print()
    # turn feature_list into a dict with the index as value -> random access
    feature_dict = {feature_list[i]: i for i in range(0, len(feature_list))}
    print(F"Index for \'people\': {feature_dict['people']}")

    NEU_LEN, LIB_LEN, CON_LEN = 14846, 4448, 4448
    ROW_LEN = M_SIZE[0]
    UNI_FACTOR, BI_FACTOR, TRI_FACTOR = 5, 1.5, .75
    VEC_ID = F"{UNI_FACTOR}{BI_FACTOR}{TRI_FACTOR}"
    print('\nIntegrating IBC data...')
    DO_IBC_INTEGRATION = True
    if DO_IBC_INTEGRATION:
        with open("./../Dataset/ibc_data/feature_lists/neu_list.csv", 'r') as f:
            reader = csv.DictReader(f)
            pbar = tqdm(total=NEU_LEN)
            for row in reader:
                if row['gram'] == '1':
                    if row['1st'] in feature_dict:
                        i = feature_dict[row['1st']]
                        for doc_i in range(ROW_LEN):
                            if text_counts[doc_i, i] > 0:
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / UNI_FACTOR
                if row['gram'] == '2':
                    word = F"{row['1st']} {row['2nd']}"
                    if word in feature_dict:
                        i = feature_dict[word]
                        for doc_i in range(ROW_LEN):
                            if text_counts[doc_i, i] > 0:
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / BI_FACTOR
                if row['gram'] == '3':
                    word = F"{row['1st']} {row['2nd']} {row['3rd']}"
                    if word in feature_dict:
                        i = feature_dict[word]
                        for doc_i in range(ROW_LEN):
                            if text_counts[doc_i, i] > 0:
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / TRI_FACTOR
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
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / UNI_FACTOR
                if row['gram'] == '2':
                    word = F"{row['1st']} {row['2nd']}"
                    if word in feature_dict:
                        i = feature_dict[word]
                        for doc_i in range(ROW_LEN):
                            if (data['allsides_bias'][doc_i] == "From the Left"
                                    and text_counts[doc_i, i] > 0):
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / BI_FACTOR
                if row['gram'] == '3':
                    word = F"{row['1st']} {row['2nd']} {row['3rd']}"
                    if word in feature_dict:
                        i = feature_dict[word]
                        for doc_i in range(ROW_LEN):
                            if (data['allsides_bias'][doc_i] == "From the Left"
                                    and text_counts[doc_i, i] > 0):
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / TRI_FACTOR

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
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / UNI_FACTOR
                if row['gram'] == '2':
                    word = F"{row['1st']} {row['2nd']}"
                    if word in feature_dict:
                        i = feature_dict[word]
                        for doc_i in range(ROW_LEN):
                            if (data['allsides_bias'][doc_i] == "From the Right"
                                    and text_counts[doc_i, i] > 0):
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / BI_FACTOR
                if row['gram'] == '3':
                    word = F"{row['1st']} {row['2nd']} {row['3rd']}"
                    if word in feature_dict:
                        i = feature_dict[word]
                        for doc_i in range(ROW_LEN):
                            if (data['allsides_bias'][doc_i] == "From the Right"
                                    and text_counts[doc_i, i] > 0):
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / TRI_FACTOR
                pbar.update(1)
            pbar.close()

        filename = F'Vectorizers/{VEC_ID}_cv.sav'
        pickle.dump(text_counts, open(filename, 'wb'))
    else:
        filename = F'Vectorizers/{VEC_ID}_cv.sav'
        text_counts = pickle.load(open(filename, 'rb'))
    print(F"Ex. features in first 5 docs:")
    for i in range(5):
        for j in range(DISPLAY_INDEX, DISPLAY_INDEX + 5):
            print(F"\t{text_counts[i, j]}", end=" ")
        print()

    print('\nTfidf transform...')
    DO_TFIDF_INTEGRATION = True
    if DO_TFIDF_INTEGRATION:
        tfidf_counts = TfidfTransformer().fit_transform(text_counts)
        filename = F'Vectorizers/{VEC_ID}_tfidf.sav'
        pickle.dump(tfidf_counts, open(filename, 'wb'))
    else:
        filename = F'Vectorizers/{VEC_ID}_tfidf.sav'
        tfidf_counts = pickle.load(open(filename, 'rb'))

    # tfidf_counts = text_counts
    print(F"Ex. features in first 5 docs:")
    for i in range(5):
        for j in range(DISPLAY_INDEX, DISPLAY_INDEX + 5):
            print(F"\t{tfidf_counts[i, j]:.2}", end=" ")
        print()

    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_counts, data['allsides_bias'], test_size=0.3, random_state=328)

    print(F"\nTraining set:")
    print(F"Ex. features in first 5 docs:")
    for i in range(5):
        for j in range(DISPLAY_INDEX, DISPLAY_INDEX + 5):
            print(F"\t{X_train[i, j]:.2}", end=" ")
        print()
    print(F"First 5 tags: \n{y_train[:5]}")

    print('\nTraining Classifier...')
    # clf = SGDClassifier().fit(X_train, y_train)
    clf = AdaBoostClassifier().fit(X_train, y_train)
    name = "AdaBoostClassifier"
    y_pred = clf.predict(X_test)
    log_new()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(F"{accuracy:.2%} - ibc{name}")
    log_results(F"{accuracy:.2%} - ibc{name}")
    my_tags = ['From the Right', 'From the Left', 'From the Center']
    print(classification_report(y_test, y_pred, target_names=my_tags))

    filename = F'Models/{accuracy:.2%}_ibc_{name}.sav'
    pickle.dump(clf, open(filename, 'wb'))

    clf = SGDClassifier().fit(X_train, y_train)
    name = "SGDClassifier"
    y_pred = clf.predict(X_test)
    log_new()
    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(F"{accuracy:.2%} - ibc{name}")
    log_results(F"{accuracy:.2%} - ibc{name}")
    my_tags = ['From the Right', 'From the Left', 'From the Center']
    print(classification_report(y_test, y_pred, target_names=my_tags))

    filename = F'Models/{accuracy:.2%}_ibc_{name}.sav'
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
