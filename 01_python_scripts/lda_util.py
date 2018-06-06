#from article_2_vector_word_count import *
#from collections import defaultdict
import lda
import sqlite3
import numpy as np
#from scipy.sparse import csr_matrix, save_npz
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame
#from NYSE_tradingdays import NYSE_tradingdays
from datetime import datetime
import pandas as pd
import pickle as pk

def date_convert(x):
        if x is None:
                return(None)
        else:
                return(datetime.strptime(x, '%Y-%m-%d').date() if len(x)<=10 else datetime.strptime(x, '%Y-%m-%d %H:%M:%S').date())

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN    
    
def is_float(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

class LemmaTokenizer(object):
     def __init__(self):
         self.wnl = WordNetLemmatizer()
     def __call__(self, doc):
         self.word_pos=pos_tag(word_tokenize(doc))
         return [self.wnl.lemmatize(w,get_wordnet_pos(p)) for w,p in self.word_pos 
                 if len(w)>=3 and not w.isdigit() and not is_float(w) ]


# -----------------------------------
# Extracting features from database
# -----------------------------------
def article_extractor(sqlite_file,start_date, end_date): 
    conn=sqlite3.connect(sqlite_file)
    c=conn.cursor()
    articles_2016=c.execute("SELECT date, article FROM articles WHERE date BETWEEN ? AND ?", (start_date, end_date))
    df=DataFrame(articles_2016.fetchall())
    df.columns=('date','article')
    conn.close()
    return df

def article2matrix(article):
    cnt_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(),stop_words='english')
    doc_term_mat = cnt_vectorizer.fit_transform(article)
    vocab = cnt_vectorizer.vocabulary_.keys()
    return (doc_term_mat, vocab)

def matrix_dump(mat, file):
    with open(file,'wb+') as f:
        pk.dump(mat,f)
        
def vocab_write(vocab, file):
    with open(file,'w+') as f:
        for wd in vocab:
            f.write(wd + '\n')    


def lda_out(doc_term_mat, vocab, directory, outfile_prefix, num_topics, n_top_words): 
    model=lda.LDA(n_topics=num_topics,n_iter=1500,random_state=1)
    model.fit(doc_term_mat)
    topic_word=model.topic_word_
    
    matrix_dump(topic_word, directory, outfile_prefix, 'tpc_wds' )
    matrix_dump(model.doc_topic_, directory, outfile_prefix, 'doc_tpc')
    
    with open(directory+outfile_prefix+str(num_topics)+'_120501_120531.txt','w+') as f:	 
        for i, topic_dist in enumerate(topic_word):
            topic_words=np.array(sorted(vocab))[np.argsort(topic_dist)][:-(n_top_words+1):-1]
            f.write('Topic {0} : {1}\n'.format(i, ', '.join(topic_words).encode("utf-8")))

def matrix_load(file):
    with open(file,'rb') as f:
        mat = pk.load(f)
    return mat

def vocab_load(file):
    with open(file, 'r') as f:
        voc=f.read()
    return voc


