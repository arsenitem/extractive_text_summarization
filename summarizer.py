#! /usr/bin/env python
# -*- coding: utf-8 -*-
from os import listdir
import math
import sys
import operator
import string
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
porter = PorterStemmer()
import pymorphy2
from heapq import nlargest
morph = pymorphy2.MorphAnalyzer()

def tokenize_ru(file_text):
    tokens = word_tokenize(file_text)

    tokens = [i for i in tokens if (i not in string.punctuation)]

    stop_words = stopwords.words('russian')
    stop_words.extend(['что', 'это', 'так', 'вот', 'быть', 'как', 'в', '—', '–', 'к', 'на', '...', ''])
    tokens = [i for i in tokens if (i not in stop_words)]

    tokens = [i.replace("«", "").replace("»", "") for i in tokens]

    return tokens

def normalize_tokens(tokens):
    normalized_tokens= []
    if sys.argv[3] == "v1":
        for token in tokens:
            normalized = morph.parse(token)[0].normal_form
            normalized_tokens.append(normalized)
    else:
        for token in tokens:   
            normalized = porter.stem(token)
            normalized_tokens.append(normalized)
    return normalized_tokens

def fill_dict(tokens):
    dictionary = {}
    for word in tokens:
        if word in dictionary:
            dictionary[word] = dictionary[word]+1
        else:
            dictionary[word] = 1
    return dictionary

def write_to_file(dictionary,file):
    for i in dictionary:
        file.write(str(i) + "\n")

def sort_dict(dictionary):
    return sorted(dictionary.items(), key=operator.itemgetter(1))

def load_corpus(path):
    print("loading document corpus...")
    result = []
    for doc in listdir(path):
        doc_path = path + "/" + doc
        corpus_file = open(doc_path, "r", encoding='utf-8')
        raw_text = corpus_file.read()
        corpus_file.close()
        tokens = tokenize_ru(raw_text)
        normalized_tokens = normalize_tokens(tokens)
        result.append(normalized_tokens)
    return result
# Количество раз, когда термин а встретился в тексте / количество всех слов в тексте
def compute_tf(dictionary, tokens):
    tf_token = {}
    for key in dictionary.keys():
        tf_token[key] = dictionary[key]/len(tokens)
    return tf_token

def compute_idf(word, corpus):
    # IDF термина а = логарифм(Общее количество документов / Количество документов, в которых встречается термин а
    res = math.log10(len(corpus)/sum([1.0 for i in corpus if word in i]))
    return res

# Это простой и удобный способ оценить важность термина для какого-либо документа относительно всех остальных документов.
# Принцип такой — если слово встречается в каком-либо документе часто,
# при этом встречаясь редко во всех остальных документах — это слово имеет большую значимость для того самого документа.
def compute_tf_idf(dictionary, tokens, corpus):
    print("computing tf-idf...")
    tf_dict = compute_tf(dictionary, tokens)
    tf_idf_dict = {}
    for key in tf_dict.keys():
        tf_dict[key]
        tf_idf_dict[key] = tf_dict[key]*compute_idf(key, corpus)
    return tf_idf_dict

def tokenize_sentences(text):
    print("tokenizing sentence...")
    tokenized_sent = []
    match_sent = {}
    for sent in sent_tokenize(text, 'russian'):
        tokens = tokenize_ru(sent)
        normalized_tokens = normalize_tokens(tokens)
        tokenized_sent.append(normalized_tokens)
        match_sent[str(normalized_tokens)] = sent
    return tokenized_sent, match_sent

def compute_sentences_score(sent_tokens, tf_idf):
    print("computing sentence score...")
    score = {}
    for sent in sent_tokens:
        for word in sent:
            if word in tf_idf.keys():
                if word in dictionary:
                    score[str(sent)] =  tf_idf[word]
                else:
                    score[str(sent)] =  tf_idf[word]
    return score

file = open(sys.argv[1], "r", encoding='utf-8')
result_file = open(sys.argv[2], "w", encoding='utf-8')

raw_text = file.read()
file.close()

tokens = tokenize_ru(raw_text)
normalized_tokens = normalize_tokens(tokens)

dictionary = fill_dict(normalized_tokens)

sorted_dictionary = sort_dict(dictionary)
write_to_file(sorted_dictionary, result_file)

corpus = load_corpus("corpus")
tf_idf = compute_tf_idf(dictionary, normalized_tokens, corpus)

sorted_tf_idf = sort_dict(tf_idf)
write_to_file(sorted_tf_idf, result_file)

sent_tokens, match_sent = tokenize_sentences(raw_text)
sent_scores = compute_sentences_score(sent_tokens,tf_idf)

sorted_sent_scores = sort_dict(sent_scores)
write_to_file(sorted_sent_scores, result_file)

compression = float(sys.argv[4])
select_length = int(len(match_sent)* compression)
summary = nlargest(select_length, sent_scores)

result_summary_array = [match_sent[item] for item in summary]
write_to_file(result_summary_array, result_file)
result_file.close()

result_summary_text = (' ').join(result_summary_array)
print("done")
print("before: "+ str(len(raw_text)) + " chars")
print("after: " + str(len(result_summary_text)) + " chars")