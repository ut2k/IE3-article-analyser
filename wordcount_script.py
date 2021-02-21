import nltk

article_full_texts = ["a","b","a","b","a","b","b","a","a"]
article_political_bias = ["bias","neutral","bias","neutral","bias","neutral","neutral","bias","bias"]
article_allsides = []

# Append all biased article texts to one long string.
# Same for all neutral articles.
biased_texts = []
neutral_texts = []

for article, bias in zip(article_full_texts, article_political_bias):
    if bias == "bias":
        biased_texts.append(article)
    elif bias == "neutral":
        neutral_texts.append(article)

biased_string = " ".join(biased_texts)
neutral_string = " ".join(neutral_texts)

print(biased_string)
print(neutral_string)

# Tokenize the biased_texts string by word.
# Tokenize the unbiased_texts string by word.
from nltk.tokenize import word_tokenize
bias_tokenized = word_tokenize(biased_string)
neutral_tokenized = word_tokenize(neutral_string)

# Remove stopwords, punctuation, and misc characters.
# -- This is incomplete, add more later.
punctuation = [",",".","!",";",":"]

from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))

filtered_bias = []
for w in bias_tokenized:
    if w not in stop_words and w not in punctuation:
        filtered_bias.append(w)

filtered_neutral = []
for w in neutral_tokenized:
    if w not in stop_words and w not in punctuation:
        filtered_neutral.append(w)

# Stem the filtered lists
from nltk.stem import PorterStemmer

ps = PorterStemmer()

stemmed_bias = []
for w in filtered_bias:
    stemmed_bias.append(ps.stem(w))

stemmed_neutral = []
for w in filtered_neutral:
    stemmed_neutral.append(ps.stem(w))

# Get wordcounts for each list and print the 20 most common ones (20 is arbitrary)
from nltk.probability import FreqDist
fdist_bias = FreqDist(stemmed_bias)
fdist_neutral = FreqDist(stemmed_neutral)
keywords_bias = fdist_bias.most_common(20)
keywords_neutral = fdist_neutral.most_common(20)

