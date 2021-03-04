# system
import pickle

# lib
import dill
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk import ngrams
import re


def get_ibc_data(use_neutral=False,
                 use_subsampling=False,
                 return_subsampling_depths=False,
                 labels=(0, 1, 2)):
    """
    Samples the annotated IBC data. This dataset can be described
    more here. http://www.cs.umd.edu/~miyyer/ibc/.
    This dataset has sub-sentance labeled political sentiment
    data. Structured like (for example),
        (The affordable care act is great for americans, liberal)
                           |
                         |   |
                        |     |
                       |       |
                      N/A   (is great for americans, concervative)
                                     |
                                   |   |
                                 |       |
                                N/A   (for americans, neutral)
    You can get the fully annotated samples (with subsamples) by givinng
    use_subsampling as true.
    The data is returned as two lists. X is uncleaned text. Y is the label.
    If using subsampling, you can also choose to get the tree depth of the sample.
    Because the annotations run along a single path, there will only be one
    sample per depth per sentence.
    You can choose whhat labels you want by giving the labels parameter
    as a list or tuple of labels in the order of (liberal, conservative, neutral).
    The default is 0,1, or 2.
    :param use_neutral: whether or not to use neutral in the data
    :param use_subsampling: whether or not to subsample (give all parts of tree)
    :param return_subsampling_depths: whether or not to return the sampling depths
    :param labels:
    :return: X, Y, P?
    """

    lib, con, neutral = pickle.load(open('../Dataset/ibc_data/ibcData.pkl', 'rb'))
    # print(len(lib))
    # print(len(con))

    if not use_neutral:
        neutral = []

    if not use_subsampling:

        extract = lambda x: [e.get_words() for e in x]
        lib, con, neutral = extract(lib), extract(con), extract(neutral)

        X = [*lib, *con, *neutral]
        Y = [*[labels[0]] * len(lib),
             *[labels[1]] * len(con),
             *[labels[2]] * len(neutral)]

        if return_subsampling_depths:
            raise ValueError('there is no subsampling positions if'
                             'this function is called without sub'
                             'sampling')

        return X, Y

    else:

        label_map = {
            'liberal': labels[0],
            'conservative': labels[1],
            'neutral': labels[2]
        }

        X, Y, P = [], [], []
        entries = [*lib, *con, *neutral]
        for tree in entries:
            depth = 0
            for node in tree:
                if hasattr(node, 'label'):
                    depth += 1
                    if use_neutral or node.label.lower() != 'neutral':
                        X.append(node.get_words())
                        Y.append(label_map[node.label.lower()])
                        P.append(depth)

        if return_subsampling_depths:
            return X, Y, P

        return X, Y


if __name__ == '__main__':

    X, Y, P = get_ibc_data(use_neutral=True,
                           use_subsampling=True,
                           return_subsampling_depths=True)

    lib_Fdist = FreqDist()
    con_Fdist = FreqDist()
    neu_Fdist = FreqDist()

    lib_dict = dict()
    con_dict = dict()
    neu_dict = dict()
    gen_dict = dict()

    MAX_FEATURES = 1500


    def find_most_common(lib_Fdist, con_Fdist, neu_Fdist, n):
        lib_len = 2025
        con_len = 1701
        neu_len = 600

        lib_Fdist.clear()
        con_Fdist.clear()
        neu_Fdist.clear()

        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        STOPWORDS = set(stopwords.words('english'))
        STOPWORDS.update(['would', 'use', 'make'])

        def clean_text(text):
            text = text.lower()  # lowercase text
            text = REPLACE_BY_SPACE_RE.sub(' ', text)  # replace REPLACE_BY_SPACE_RE symbols by space in text
            text = BAD_SYMBOLS_RE.sub('', text)  # delete symbols which are in BAD_SYMBOLS_RE from text
            text = ' '.join(word for word in text.split() if word not in STOPWORDS)  # delete stopwors from text
            return text

        i = 0
        # nomalize by number of sentences
        for sent in X:
            clean_sent = clean_text(sent)
            for g in ngrams(word_tokenize(clean_sent), n):
                if Y[i] == 0:
                    lib_Fdist[g] += 1000 / lib_len
                if Y[i] == 1:
                    con_Fdist[g] += 1000 / con_len
                if Y[i] == 2:
                    neu_Fdist[g] += 1000 / neu_len

            i += 1

    print("\nwriting to memory...")

    def fdist_to_dict(fd, d):
        print(fd.most_common(5))
        for term, freq in fd.most_common(MAX_FEATURES): # for n-grams: [(("word_1","word_2",...,"word_n"), freq),...]
            d.setdefault(term, []).append(freq)

    find_most_common(lib_Fdist, con_Fdist, neu_Fdist, 1)
    fdist_to_dict(lib_Fdist, lib_dict)
    fdist_to_dict(con_Fdist, con_dict)
    fdist_to_dict(neu_Fdist, neu_dict)
    print()
    find_most_common(lib_Fdist, con_Fdist, neu_Fdist, 2)
    fdist_to_dict(lib_Fdist, lib_dict)
    fdist_to_dict(con_Fdist, con_dict)
    fdist_to_dict(neu_Fdist, neu_dict)
    print()
    find_most_common(lib_Fdist, con_Fdist, neu_Fdist, 3)
    fdist_to_dict(lib_Fdist, lib_dict)
    fdist_to_dict(con_Fdist, con_dict)
    fdist_to_dict(neu_Fdist, neu_dict)

    # create neu_list.csv from gen_dict
    def add_to_gen_dict(d):
        for term, freq in d.items():
            if term in gen_dict:
                # gen_dict[term][0] += freq[0]
                temp = gen_dict[term][0] + freq[0]
                gen_dict.update({term: [temp]})
            else:
                gen_dict[term] = freq

    add_to_gen_dict(lib_dict)
    add_to_gen_dict(con_dict)
    add_to_gen_dict(neu_dict)

    print("\nwriting to files...")
    def dict_to_file(path, d):
        with open(path, 'w') as f:
            f.write("gram,1st,2nd,3rd,freq\n")
            for term, freq in d.items():
                if len(term) == 1:
                    f.write(F"1,{term[0]}, , ,{freq[0]}\n")
                if len(term) == 2:
                    f.write(F"2, {term[0]},{term[1]}, ,{freq[0]}\n")
                if len(term) == 3:
                    f.write(F"3, {term[0]},{term[1]},{term[2]},{freq[0]}\n")

    dict_to_file("../Dataset/ibc_data/feature_lists/neu_list.csv", gen_dict)

    # remove all overlap between lib and conv
    temp_lib_dict = dict()
    for term, lib_freq in lib_dict.items():
        if term in con_dict:  # only keep the more frequent one
            con_dict.pop(term)
        else:
            temp_lib_dict[term] = lib_freq

    lib_dict = temp_lib_dict

    dict_to_file("../Dataset/ibc_data/feature_lists/lib_list.csv", lib_dict)
    dict_to_file("../Dataset/ibc_data/feature_lists/con_list.csv", con_dict)


