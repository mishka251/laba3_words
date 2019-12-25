import codecs

import matplotlib
import pandas as pd
from nltk import RegexpTokenizer
from pandas import DataFrame
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

import matplotlib.pyplot as plt
import numpy as np


def read_file(file_name: str, clean_file_name: str, target_column: str) -> DataFrame:
    output_file = open(clean_file_name, "w", encoding='utf-8')

    def sanitize_characters(raw, clean):
        for line in input_file:
            out = line
            output_file.write(line)

    input_file = codecs.open(file_name, "r", encoding='utf-8', errors='replace')
    sanitize_characters(input_file, output_file)

    questions = pd.read_csv(clean_file_name)
    # questions.columns = ['text', 'tag']
    questions.head()

    questions.describe()

    def standardize_text(df: DataFrame, text_field: str):
        df[text_field] = df[text_field].str.replace(r"http\S+", "")
        df[text_field] = df[text_field].str.replace(r"http", "")
        df[text_field] = df[text_field].str.replace(r"@\S+", "")
        df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
        df[text_field] = df[text_field].str.replace(r"@", "at")
        df[text_field] = df[text_field].str.lower()
        return df

    questions = standardize_text(questions, "text")
    questions.to_csv("clean_data.csv")
    questions.head()

    _clean_questions: DataFrame = pd.read_csv("clean_data.csv")
    _clean_questions.tail()

    _clean_questions.groupby(target_column).count()
    _clean_questions[target_column] = _clean_questions[target_column].apply(lambda s: "1" if s == "pos" else "0")

    _tokenizer: RegexpTokenizer = RegexpTokenizer(r'\w+')
    _clean_questions["tokens"] = _clean_questions["text"].apply(_tokenizer.tokenize)
    _clean_questions.head()

    return _clean_questions


def plot_LSA(test_data, test_labels, savepath="PCA_demo.csv", use_plot=True, mpatches=None):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ['orange', 'blue', 'blue']
    if use_plot:
        plt.scatter(lsa_scores[:, 0], lsa_scores[:, 1], s=8, alpha=.8, c=test_labels,
                    cmap=matplotlib.colors.ListedColormap(colors))
        red_patch = mpatches.Patch(color='orange', label='Irrelevant')
        green_patch = mpatches.Patch(color='blue', label='Film')
        plt.legend(handles=[red_patch, green_patch], prop={'size': 30})


def get_metrics(_y_test, y_predicted):
    # true positives / (true positives+false positives)
    _precision = precision_score(_y_test, y_predicted, pos_label=None, average='weighted')
    # true positives / (true positives + false negatives)
    _recall = recall_score(_y_test, y_predicted, pos_label=None, average='weighted')
    # harmonic mean of precision and recall
    _f1 = f1_score(_y_test, y_predicted, pos_label=None, average='weighted')
    # true positives + true negatives/ total
    _accuracy = accuracy_score(_y_test, y_predicted)
    return _accuracy, _precision, _recall, _f1


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.winter):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=20)
    plt.yticks(tick_marks, classes, fontsize=20)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i, j in np.itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black", fontsize=40)
    plt.tight_layout()
    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    return plt


def get_most_important_features(vectorizer, model, n=5):
    index_to_word = {v: k for k, v in vectorizer.vocabulary_.items()}
    # loop for each class
    classes = {}
    for class_index in range(model.coef_.shape[0]):
        word_importances = [(el, index_to_word[i]) for i, el in enumerate(model.coef_[class_index])]
        sorted_coeff = sorted(word_importances, key=lambda x: x[0], reverse=True)
        tops = sorted(sorted_coeff[:n], key=lambda x: x[0])
        bottom = sorted_coeff[-n:]
        classes[class_index] = {
            'tops': tops,
            'bottom': bottom
        }
    return classes


def plot_important_words(top_scores, top_words, bottom_scores, bottom_words, name):
    y_pos = np.arange(len(top_words))
    top_pairs = [(a, b) for a, b in zip(top_words, top_scores)]
    top_pairs = sorted(top_pairs, key=lambda x: x[1])
    bottom_pairs = [(a, b) for a, b in zip(bottom_words, bottom_scores)]
    bottom_pairs = sorted(bottom_pairs, key=lambda x: x[1], reverse=True)
    top_words = [a[0] for a in top_pairs]
    top_scores = [a[1] for a in top_pairs]
    bottom_words = [a[0] for a in bottom_pairs]
    bottom_scores = [a[1] for a in bottom_pairs]
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(121)
    plt.barh(y_pos, bottom_scores, align='center', alpha=0.5)
    plt.title('Irrelevant', fontsize=20)
    plt.yticks(y_pos, bottom_words, fontsize=14)
    plt.suptitle('Key words', fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    plt.subplot(122)
    plt.barh(y_pos, top_scores, align='center', alpha=0.5)
    plt.title('Film', fontsize=20)
    plt.yticks(y_pos, top_words, fontsize=14)
    plt.suptitle(name, fontsize=16)
    plt.xlabel('Importance', fontsize=20)
    plt.subplots_adjust(wspace=0.8)
    plt.show()


def plot_top_scores(importances):
    # print(importance)
    top_scores = [a[0] for a in importances[0]['tops']]
    top_words = [a[1] for a in importances[0]['tops']]
    bottom_scores = [a[0] for a in importances[0]['bottom']]
    bottom_words = [a[1] for a in importances[0]['bottom']]
    plot_important_words(top_scores, top_words, bottom_scores, bottom_words, "Most important words for relevance")
