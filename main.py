from typing import List, Tuple, Any

# import keras
# import nltk
# import pandas as pd
import numpy as np
# import re
# import codecs
#
# from sklearn.decomposition import PCA, TruncatedSVD
# import matplotlib
# import matplotlib.patches as mpatches
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# from keras.utils import to_categorical
# from nltk.tokenize import RegexpTokenizer

import matplotlib.pyplot as plt
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

# import numpy as np
# import itertools
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.linear_model import LogisticRegression

from keras.layers import Dense, Input, Flatten, Dropout, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding
# from keras.layers import LSTM, Bidirectional
from keras.models import Model

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from pandas import DataFrame

import gensim

from utils import read_file, plot_LSA, get_metrics, plot_confusion_matrix, get_most_important_features, plot_top_scores

input_file_name: str = "movie_review.csv"  # "movie_review.csv"#"socialmedia_relevant_cols.csv"
clean_file_name: str = "clean_" + input_file_name

target_column: str = "tag"  # "class_label"#"tag"

clean_questions: DataFrame = read_file(input_file_name, clean_file_name, target_column)
all_words: List[str] = [word for tokens in clean_questions["tokens"] for word in tokens]
sentence_lengths: List[int] = [len(tokens) for tokens in clean_questions["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))

fig = plt.figure(figsize=(10, 10))
plt.xlabel('Sentence length')
plt.ylabel('Number of sentences')
plt.hist(sentence_lengths)
plt.show()


def cv(data: DataFrame) -> Tuple[Any, CountVectorizer]:
    _count_vectorizer = CountVectorizer()
    emb = _count_vectorizer.fit_transform(data)
    return emb, _count_vectorizer


list_corpus: List = clean_questions["text"].tolist()
list_labels: List = clean_questions[target_column].tolist()
X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, random_state=40)
X_train_counts, count_vectorizer = cv(X_train)


def process_train_result(train_counts, count_vecrorizer):
    X_test_counts = count_vectorizer.transform(X_test)

    fig = plt.figure(figsize=(16, 16))

    plot_LSA(X_train_counts, y_train)
    plt.show()

    clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial', n_jobs=-1,
                             random_state=40)
    clf.fit(X_train_counts, y_train)
    y_predicted_counts = clf.predict(X_test_counts)

    accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
    print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

    cm = confusion_matrix(y_test, y_predicted_counts)
    fig = plt.figure(figsize=(10, 10))
    plot = plot_confusion_matrix(cm, classes=['Irrelevant', 'Film', 'Unsure'], normalize=
    False, title='Confusion matrix')
    plt.show()
    return cm, clf


cm, clf = process_train_result(X_train_counts, count_vectorizer)
print(cm)

importance = get_most_important_features(count_vectorizer, clf, 10)

plot_top_scores(importance)


def tfidf(data):
    tfidf_vectorizer = TfidfVectorizer()
    train = tfidf_vectorizer.fit_transform(data)
    return train, tfidf_vectorizer


X_train_tfidf, tfidf_vectorizer = tfidf(X_train)
cm2, clf_tfidf = process_train_result(X_train_tfidf, tfidf_vectorizer)

# X_test_tfidf = tfidf_vectorizer.transform(X_test)
# fig = plt.figure(figsize=(16, 16))
# plot_LSA(X_train_tfidf, y_train)
# plt.show()
#
# clf_tfidf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
#                                multi_class='multinomial', n_jobs=-1, random_state=40)
# clf_tfidf.fit(X_train_tfidf, y_train)
# y_predicted_tfidf = clf_tfidf.predict(X_test_tfidf)
#
# accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf = get_metrics(y_test, y_predicted_tfidf)
# print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (
#     accuracy_tfidf, precision_tfidf, recall_tfidf, f1_tfidf))
#
# cm2 = confusion_matrix(y_test, y_predicted_tfidf)
# fig = plt.figure(figsize=(10, 10))
# plot = plot_confusion_matrix(cm2, classes=['Irrelevant', 'Film', 'Unsure'], normalize
# =False, title='Confusion matrix')
# plt.show()
print("TFIDF confusion matrix")
print(cm2)
print("BoW confusion matrix")
print(cm)

importance_tfidf = get_most_important_features(tfidf_vectorizer, clf_tfidf, 10)
plot_top_scores(importance_tfidf)
# top_scores = [a[0] for a in importance_tfidf[1]['tops']]
# top_words = [a[1] for a in importance_tfidf[1]['tops']]
# bottom_scores = [a[0] for a in importance_tfidf[1]['bottom']]
# bottom_words = [a[1] for a in importance_tfidf[1]['bottom']]
# plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")

word2vec_path = "GoogleNews-vectors-negative300.bin.gz"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)


def get_average_word2vec(tokens_list, vector, generate_missing=False, k=300):
    if len(tokens_list) < 1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]

    length = len(vectorized)
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged


def get_word2vec_embeddings(vectors, clean_questions, generate_missing=False):
    embeddings = clean_questions['tokens'].apply(
        lambda x: get_average_word2vec(x, vectors, generate_missing=generate_missing))
    return list(embeddings)


embeddings = get_word2vec_embeddings(word2vec, clean_questions)
X_train_word2vec, X_test_word2vec, y_train_word2vec, y_test_word2vec = train_test_split(embeddings, list_labels,
                                                                                        test_size=0.2, random_state=40)

fig = plt.figure(figsize=(16, 16))
plot_LSA(embeddings, list_labels)
plt.show()

clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial',
                             random_state=40)
clf_w2v.fit(X_train_word2vec, y_train_word2vec)
y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)

clf_w2v = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', multi_class='multinomial',
                             random_state=40)
clf_w2v.fit(X_train_word2vec, y_train_word2vec)
y_predicted_word2vec = clf_w2v.predict(X_test_word2vec)

cm_w2v = confusion_matrix(y_test_word2vec, y_predicted_word2vec)
fig = plt.figure(figsize=(10, 10))
plot = plot_confusion_matrix(cm, classes=['Irrelevant', 'Film', 'Unsure'], normalize=False,
                             title='Confusion matrix')
plt.show()
print("Word2Vec confusion matrix")
print(cm_w2v)
print("TFIDF confusion matrix")
print(cm2)
print("BoW confusion matrix")
print(cm)

EMBEDDING_DIM = 300
MAX_SEQUENCE_LENGTH = 35
VOCAB_SIZE = len(VOCAB)
VALIDATION_SPLIT = .2
tokenizer = Tokenizer(num_words=VOCAB_SIZE)
tokenizer.fit_on_texts(clean_questions["text"].tolist())
sequences = tokenizer.texts_to_sequences(clean_questions["text"].tolist())
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
cnn_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
labels = to_categorical(np.asarray(clean_questions[target_column]))
indices = np.arange(cnn_data.shape[0])
np.random.shuffle(indices)
cnn_data = cnn_data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * cnn_data.shape[0])
embedding_weights = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, index in word_index.items():
    embedding_weights[index, :] = word2vec[word] if word in word2vec else np.random.rand(EMBEDDING_DIM)
print(embedding_weights.shape)


def ConvNet(embeddings, max_sequence_length, num_words, embedding_dim, labels_index, trainable=False, extra_conv=True):
    embedding_layer = Embedding(num_words, embedding_dim, weights=[embeddings], input_length=max_sequence_length,
                                trainable=trainable)
    sequence_input = Input(shape=(max_sequence_length,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    # Yoon Kim model (https://arxiv.org/abs/1408.5882)
    convs = []
    filter_sizes = [3, 4, 5]
    for filter_size in filter_sizes:
        l_conv = Conv1D(filters=128, kernel_size=filter_size, activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(pool_size=3)(l_conv)
        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)
    # add a 1D convnet with global maxpooling, instead of Yoon Kim model
    conv = Conv1D(filters=128, kernel_size=3, activation='relu')(embedded_sequences)
    pool = MaxPooling1D(pool_size=3)(conv)
    if extra_conv == True:
        x = Dropout(0.5)(l_merge)
    else:
        # Original Yoon Kim model
        x = Dropout(0.5)(pool)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # x = Dropout(0.5)(x)
    preds = Dense(labels_index, activation='softmax')(x)
    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


x_train = cnn_data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = cnn_data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]
model = ConvNet(embedding_weights, MAX_SEQUENCE_LENGTH, len(word_index) + 1, EMBEDDING_DIM,
                len(list(clean_questions[target_column].unique())), False)

model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=3, batch_size=128)
print("end")
