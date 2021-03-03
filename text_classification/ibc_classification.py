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

tqdm.pandas(desc="progress-bar")
import csv
from sklearn.ensemble import AdaBoostClassifier
from sklearn import metrics
from sklearn.linear_model import SGDClassifier
import time


def main():
    clean_json('../released_data.json', '../Dataset/cleaned_data.json')  # uncomment to ues full dataset
    # clean_json('../Dataset/test_article.json', '../Dataset/cleaned_data.json') # comment to use smaller dataset

    print("\nReading json...")
    # reads json data into a pandas dataframe object
    data = pd.read_json('../Dataset/cleaned_data.json')
    data.info()

    ibc_classify(data)

    return


def clean_text(text):
    REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
    BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
    STOPWORDS = set(stopwords.words('english'))

    text = text.lower()  # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
    return text


def ibc_classify(data: pd.DataFrame):
    log_new()
    print("First 5 docs:")
    for i in range(5):
        print(F"{i + 1}: {data['content'][i][:75]}...")

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

    filename = 'Features/ibc_cv.sav'
    pickle.dump(cv, open(filename, 'wb'))

    M_SIZE = text_counts.shape
    DISPLAY_INDEX = 2901508  # index for the word "people"
    I_RANGE = range(DISPLAY_INDEX, DISPLAY_INDEX + 3)
    print(F"Matrix size: {M_SIZE}")

    feature_list = cv.get_feature_names()
    print(F"Select features: {feature_list[DISPLAY_INDEX: DISPLAY_INDEX + 3]}")
    print(F"Select features in first 5 docs:")
    for i in range(5):
        for j in I_RANGE:
            print(F"\t({i}, {j}) {text_counts[i, j]}", end=" ")
        print()
    # turn feature_list into a dict with the index as value -> random access
    feature_dict = {feature_list[i]: i for i in range(0, len(feature_list))}
    print(F"Index for \'people\': {feature_dict['people']}")

    filename = 'Features/feature_dict.sav'
    pickle.dump(feature_dict, open(filename, 'wb'))

    NEU_LEN, LIB_LEN, CON_LEN = 7281, 2145, 2145
    ROW_LEN = M_SIZE[0]
    FACTORS = (0.01, .0005, .00005)
    VEC_ID = F"{FACTORS[0]}{FACTORS[1]}{FACTORS[2]}"

    print('\nIntegrating IBC data...')

    DO_IBC_INTEGRATION = False

    if DO_IBC_INTEGRATION:
        def integrate_ibc(path, bias, LEN):
            with open(path, 'r') as f:
                reader = csv.DictReader(f)
                pbar = tqdm(total=LEN)
                for row in reader:
                    num_of_words = int(row['gram'])
                    if num_of_words == 1:
                        word = row['1st']
                    elif num_of_words == 2:
                        word = F"{row['1st']} {row['2nd']}"
                    elif num_of_words == 3:
                        word = F"{row['1st']} {row['2nd']} {row['3rd']}"
                    if word in feature_dict:
                        i = feature_dict[word]
                        for doc_i in range(ROW_LEN):
                            if (bias in data['allsides_bias'][doc_i]
                                    and text_counts[doc_i, i] > 0):
                                text_counts[doc_i, i] *= 1.0 * float(row['freq']) / FACTORS[num_of_words - 1]

                    pbar.update(1)
                pbar.close()
            return

        integrate_ibc("./../Dataset/ibc_data/feature_lists/neu_list.csv",
                      "From the", NEU_LEN)
        integrate_ibc("./../Dataset/ibc_data/feature_lists/lib_list.csv",
                      "From the Left", LIB_LEN)
        integrate_ibc("./../Dataset/ibc_data/feature_lists/con_list.csv",
                      "From the Right", CON_LEN)

        filename = F'Vectorizers/{VEC_ID}_cv.sav'
        pickle.dump(text_counts, open(filename, 'wb'))
    else:
        filename = F'Vectorizers/{VEC_ID}_cv.sav'
        text_counts = pickle.load(open(filename, 'rb'))

    print(F"Select features in first 5 docs:")
    for i in range(5):
        for j in I_RANGE:
            print(F"\t({i}, {j}) {text_counts[i, j]}", end=" ")
        print()

    # print('\nTfidf transform...')
    # DO_TFIDF_INTEGRATION = False
    # if DO_TFIDF_INTEGRATION:
    #     tfidf_counts = TfidfTransformer().fit_transform(text_counts)
    #     filename = F'Vectorizers/{VEC_ID}_tfidf.sav'
    #     pickle.dump(tfidf_counts, open(filename, 'wb'))
    # else:
    #     filename = F'Vectorizers/{VEC_ID}_tfidf.sav'
    #     tfidf_counts = pickle.load(open(filename, 'rb'))
    #
    # tfidf_counts = text_counts
    # print(F"Select features in first 5 docs:")
    # for i in range(5):
    #     for j in I_RANGE:
    #         print(F"\t{float(tfidf_counts[i, j]):.2}", end=" ")
    #     print()
    RANDOM_STATE = 999
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['allsides_bias'], test_size=0.3, random_state=RANDOM_STATE)

    print(F"\nTraining set:")
    print(F"Select features in first 5 docs:")
    for i in range(5):
        for j in I_RANGE:
            print(F"\t({i}, {j}) {X_train[i, j]}", end=" ")
        print()
    print(F"First 5 tags: \n{y_train[:5]}")

    print('\nTraining Classifier...')
    # clf = SGDClassifier().fit(X_train, y_train)

    DO_CLASSIFICATION = False
    name = "AdaBoostClassifier"
    percent = "94.514%"

    start = time.time()
    if DO_CLASSIFICATION:
        clf = AdaBoostClassifier().fit(X_train, y_train)
    else:
        filename = F"Models/{percent}_ibc_{name}.sav"
        clf = pickle.load(open(filename, 'rb'))
    end = time.time()
    print(F"elapsed time: {(end - start) / 60:.3} min")

    y_pred = clf.predict(X_test)

    accuracy = metrics.accuracy_score(y_test, y_pred)
    print(F"{accuracy:.2%} - ibc{name}")
    if DO_CLASSIFICATION or DO_IBC_INTEGRATION:
        log_results(F"{accuracy:.2%} - ibc{name} - {VEC_ID} ({RANDOM_STATE})")
    my_tags = ['From the Right', 'From the Left', 'From the Center']
    print(classification_report(y_test, y_pred, target_names=my_tags))

    if DO_CLASSIFICATION:
        filename = F'Models/{accuracy:.3%}_ibc_{name}.sav'
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
    print("\nCleaning json...")
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
