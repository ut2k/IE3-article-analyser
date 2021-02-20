import json
import csv
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from sklearn.feature_extraction.text import TfidfVectorizer


def main():

    print("\nCleaning json...")
    clean_json('../released_data.json', '../Dataset/cleaned_data.json')
    # clean_json('../Dataset/small_test_data.json', '../Dataset/cleaned_data.json')

    print("\nReading json...")
    # reads json data into a pandas dataframe object
    data = pd.read_json('../Dataset/cleaned_data.json')
    data.info()
    # print(data.head(5))

    print("\nCV training...")
    count_vectorization(data)
    print("\nTF training...")
    tf_idf(data)
    return


def count_vectorization(data: pd.DataFrame):
    # generates document term matrix using count vectorizer
    token = RegexpTokenizer(r'[a-zA-Z0-9]+')
    # removes stop words, symbols and change to lowercase
    cv = CountVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 1), tokenizer=token.tokenize)
    text_counts = cv.fit_transform(data['content'])
    # split into training and testing sets with random state
    X_train, X_test, y_train, y_test = train_test_split(
        text_counts, data['allsides_bias'], test_size=0.3, random_state=1)

    # fit model on training set and use it on test set
    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(F"CountVectorizer Accuracy:  {metrics.accuracy_score(y_test, predicted): .2%}")
    return


# use tf-idf to normalize document term matrix
def tf_idf(data: pd.DataFrame):
    tf = TfidfVectorizer()
    text_tf = tf.fit_transform(data['content'])
    X_train, X_test, y_train, y_test = train_test_split(
        text_tf, data['allsides_bias'], test_size=0.3, random_state=123)
    clf = MultinomialNB().fit(X_train, y_train)
    predicted = clf.predict(X_test)
    print(F"TF-IDF Accuracy:  {metrics.accuracy_score(y_test, predicted): .2%}")

    return


def clean_json(path_r : str, path_w : str):
    with open(path_r, 'r') as f_r:
        with open(path_w, 'w') as f_w:
            count = 0;
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
