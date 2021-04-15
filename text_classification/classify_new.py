import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
import csv
import time
from tqdm import tqdm
from scipy import sparse
tqdm.pandas(desc="progress-bar")


def main():
    start = time.time()
    print("\nReading json...")
    # reads json data into a pandas dataframe object
    data = pd.read_json('../Dataset/test_article.json')
    data.info()

    ibc_classify(data)
    end = time.time()
    print(F"elapsed time: {(end - start) :.3} sec")
    return


def ibc_classify(data: pd.DataFrame):

    LEN = len(data['content'])

    filename = F"Features/ibc_cv.sav"
    cv = pickle.load(open(filename, 'rb'))

    print("\nGenerate bag of words matrix...")
    text_counts = cv.transform(tqdm(data['content']))  # returns a sparse matrix, entry = matrix[x, y]
    M_SIZE = text_counts.shape
    DISPLAY_INDEX = 2901508  # index for the word "people"
    WORD = "people"

    print(F"Matrix size: {M_SIZE}")
    print(F"Select feature: [{WORD}]")
    print(F"Select feature in docs:")
    for i in range(LEN):
        print(F"\t({i}, {DISPLAY_INDEX}) {text_counts[i, DISPLAY_INDEX]}")

    filename = 'Features/feature_dict.sav'
    feature_dict = pickle.load(open(filename, 'rb'))

    NEU_LEN, LIB_LEN, CON_LEN = 14846, 4448, 4448
    ROW_LEN = LEN
    FACTORS = (0.01, .0025, .0015)
    CLF_NAME = "AdaBoostClassifier"
    PERCENT = "93.956%"

    print('\nIntegrating IBC data...')

    def integrate_ibc(path, length, tc):
        lil_tc = sparse.lil_matrix(tc)
        with open(path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                num_of_words = int(row['gram'])
                if num_of_words == 1:
                    word = row['1st']
                elif num_of_words == 2:
                    word = F"{row['1st']} {row['2nd']}"
                elif num_of_words == 3:
                    word = F"{row['1st']} {row['2nd']} {row['3rd']}"
                if word in feature_dict:
                    word_i = feature_dict[word]
                    for doc_i in range(ROW_LEN):
                        if lil_tc[doc_i, word_i] > 0:
                            lil_tc[doc_i, word_i] *= float(row['freq']) / FACTORS[num_of_words - 1]

            return sparse.csr_matrix(lil_tc)

    text_counts =integrate_ibc("./../Dataset/ibc_data/feature_lists/neu_list.csv", NEU_LEN, text_counts)
    text_counts =integrate_ibc("./../Dataset/ibc_data/feature_lists/lib_list.csv", LIB_LEN, text_counts)
    text_counts =integrate_ibc("./../Dataset/ibc_data/feature_lists/con_list.csv", CON_LEN, text_counts)

    print(F"Select feature in docs:")
    for i in range(LEN):
        print(F"\t({i}, {DISPLAY_INDEX}) {text_counts[i, DISPLAY_INDEX]}")

    filename = F"Models/{PERCENT}_ibc_{CLF_NAME}.sav"
    clf = pickle.load(open(filename, 'rb'))

    predicted = clf.predict_proba(text_counts)
    print()
    for i in range(len(predicted)):
        # print("\t\"" + article['title'][i][:50] + "...\" : " + predicted[i]) # print class
        proba_str = F"L: {predicted[i][1]:.2%}\t| R: {predicted[i][2]:.2%}\t| C: {predicted[i][0]:.2%}"
        print(F"\"{data['title'][i][:50]}...\" \n\t" + proba_str)

    return


if __name__ == '__main__':
    main()
