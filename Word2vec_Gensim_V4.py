# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 18:01:42 2019

@author: HP
"""
#Import Libraries
import pandas as pd
import numpy as np
import string
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
nltk.download('stopwords')
import pyodbc
from bs4 import BeautifulSoup
import re

from gensim.models import Word2Vec
from gensim.models.phrases import Phraser, Phrases


##############################################
#Loading input
import pickle
with open(r'G:\Patents\Patents\Semanticwords_Input_Dataset_By_IC\Unique110\Patent_tblAbstractXMLData_AutomotiveUniq_110.pkl', 'rb') as f:
    text_list = pickle.load(f)
    
# separating titles and abstracts
titlelst = []
abstractlst = []
for id,title,abstract in text_list:
    titlelst.append(title)
    abstractlst.append(abstract)

#with open(r'C:\NewsData_Variables\Body_list2.pkl', 'rb') as f:
#    Body_list2 = pickle.load(f)

#del Body_list2

indv_lines = titlelst + abstractlst
print('Found %s texts.' % len(indv_lines))

#with open(r"complete_raw_corpus_as_list.pkl",'wb') as outfile:
#    pickle.dump(indv_lines,outfile)

#create word tokens as well as remove puntuation in one go
rem_tok_punc = RegexpTokenizer(r'\w+')

def sentence_to_wordlist(raw_text):
    tokens = rem_tok_punc.tokenize(raw_text)
    return tokens

# concatenate all sentences from all texts into a single list of sentences
all_sentences = []
for raw_text in indv_lines:
    all_sentences.append(sentence_to_wordlist(raw_text))

print(len(all_sentences))

#with open(r"complete_corpus_cleaned_as_list_of_tokens.pkl",'wb') as outfile:
#    pickle.dump(all_sentences,outfile)

# Phrase Detection
# Give some common terms that can be ignored in phrase detection
# For example, 'state_of_affairs' will be detected because 'of' is provided here: 
common_terms = ["of", "with", "without", "and", "or"] #removed words 'the' and 'a'
# Create the relevant phrases from the list of sentences:
phrases = Phrases(all_sentences, common_terms=common_terms)
# The Phraser object is used from now on to transform sentences
bigram = Phraser(phrases)
# Applying the Phraser to transform our sentences is simply
all_sentences = list(bigram[all_sentences])

#with open(r"complete_corpus_cleaned_as_list_of_uni-nd-bigrams.pkl",'wb') as outfile:
#    pickle.dump(all_sentences,outfile)
print('Model Building Started')

model = Word2Vec(all_sentences, 
                 min_count=3,   # Ignore words that appear less than this
                 size=300,      # Dimensionality of word embeddings
                 workers=8,     # Number of processors (parallelisation)
                 window=5,      # Context window for words during training
                 iter=30)       # Number of epochs training over corpus

print('Model Building  Completed')
with open(r"gensim_patents_AutomotiveUniq_model_110.pkl",'wb') as outfile:
    pickle.dump(model,outfile)
    
#model.save("gensim_news_2018_19_model.bin")

#Finding similar words
model.wv.most_similar(positive='pfizer', topn=50)


model.wv.most_similar(positive=['Apple','iPhone'],negative='Google')
