# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Stem tokens (using Porter stemmer) and insert tokens from docs into inverted index

# %%
import os
import nltk
from nltk.stem import PorterStemmer
from InvertedIndex import InvertedIndex  # Custom library
from VectorSpace import VectorSpace

docs_path = "../Dataset/docs/"
docs_list = os.listdir(docs_path)

# Used later for query preprocessing
removed_frequent_words = []
replaced_words = {}

invertedIndex = InvertedIndex()
ps = PorterStemmer()

for doc_name in docs_list:
    file_path = os.path.join(docs_path, doc_name)
    doc = open(file_path)
    for token in doc:
        invertedIndex.insert(ps.stem(token[:-1].lower()), doc_name)
    doc.close()

# %% [markdown]
# Filter inverted tokens words using stopwords and replace tokens with synonyms

# %%
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords")
stopwords = stopwords.words("english")

for word in stopwords:
    invertedIndex.delete(ps.stem(word))

# %% [markdown]
# Plot token frequency distribution in filtered inverted index

# %%
import operator
import matplotlib.pyplot as plt


def plotFrequencyDistribution():
    tokenFrequencies = {}
    for token, data in invertedIndex.getTokens():
        tokenFrequencies[token] = invertedIndex.countOccurances(token)
    tokenFrequencies = dict(
        sorted(tokenFrequencies.items(), key=operator.itemgetter(1), reverse=True)
    )
    plt.plot(
        range(len(list(tokenFrequencies.values()))), list(tokenFrequencies.values())
    )
    plt.title("Frequency Distribution")
    plt.xlabel("Token # with descending frequency")
    plt.ylabel("Frequencies")
    plt.show()
    return tokenFrequencies


tokenFrequencies = plotFrequencyDistribution()

# %% [markdown]
# As you can see, very few tokens are very frequent and the rest is very infrequent.
#
# Thus I decided to remove very frequet tokens, that appear on mostly every document, since I don't want the information retrieval system to focus on those when computing answers to queries.
#
# I also replaced infrequent tokens with synonyms, hoping that the synonyms would have a higher frequency in the docs, so that by doing the same replacements in the query tokens the answers of the infrequent tokens will be grouped with their synonyms.
#
# I set the thresholds for the frequent token removal and the infrequent token replacement so that when later plotting the Frequency Distribution graph again, I will only have the "knee" of the currrent graph. This is an emperical rule to make sure that we don't have many very frequent or very infrequent tokens in the inverted index

# %%
from nltk.corpus import wordnet

nltk.download("wordnet")

# Remove frequent tokens
for token, frequency in tokenFrequencies.items():
    if frequency > 100:
        invertedIndex.delete(token)
        removed_frequent_words.append(token)
    else:
        break

# Replace infrequent words with synonyms
for token, frequency in reversed(tokenFrequencies.items()):
    if frequency < 15:
        synonyms = wordnet.synsets(token)
        for synonym in synonyms:
            new_token = ps.stem(synonym.lemmas()[0].name())  # type: ignore
            if (
                new_token != token
                and "_" not in new_token
                and new_token in invertedIndex.getTokens()
            ):
                replaced_words[token] = new_token
                invertedIndex.rename(token, new_token)
                break
    else:
        break

# %% [markdown]
# Now i plot the frequency distribution again in order to evaluate what it looks like

# %%
tokenFrequencies = plotFrequencyDistribution()

# %% [markdown]
# It is visible now that we are left with the "knee" of the original frequency distribution, so all very frequent and all very infrequent tokens are removed.
# Now let's enter the documents and queries into the Vector Space Model:

# %%
queries = open("../Dataset/Queries_20")
vectorSpace = VectorSpace(
    documents=docs_list,
    queries=queries,
    invertedIndex=invertedIndex,
    document_tf_weighting=VectorSpace.tf_simplefrequency,
    document_idf_weighting=VectorSpace.idf_logsimple,
    document_normalize_function=VectorSpace.cosine_normalization,
    query_tf_weighting=VectorSpace.tf_doublehalfnormalization,
    query_idf_weighting=VectorSpace.idf_logsimple,
    query_normalize_function=VectorSpace.no_normalization,
    query_stopwords=stopwords,
    query_removed_frequent_words=removed_frequent_words,
    query_replaced_words=replaced_words,
    query_stemmer=ps,
)
queries.close()

# %%
print(vectorSpace.lookup('0', 10))
