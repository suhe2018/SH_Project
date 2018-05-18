#!usr/bin/env python
# -*- coding:utf-8 -*-
"""
Created on Sat Sep  1 21:39:20 2017
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import random
import jieba
import numpy as np
from six.moves import xrange
import tensorflow as tf
from time import time
import  gensim
import multiprocessing

from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
import logging
import multiprocessing

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)
# Step 1: Download the data.
# Read the data into a list of strings.
def read_data():
    """
    对要训练的文本进行处理，最后把文本的内容的所有词放在一个列表中
    """
    #读取停用词
    stop_words = []
    with open('./data/stop_words.txt',"r",encoding="UTF-8") as f:
        line = f.readline()
        while line:
            # line=line.encode("utf-8")
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))

    # 读取文本，预处理，分词，得到词典
    raw_word_list = []
    with open('./data/corpus.txt',"r", encoding='UTF-8') as f:
        line = f.readline()
        while line:
            while '\n' in line:
                line = line.replace('\n','')
            while ' ' in line:
                line = line.replace(' ','')
            if len(line)>0: # 如果句子非空
                raw_words = list(jieba.cut(line,cut_all=False))
                raw_word_list.extend(raw_words)
            line=f.readline()
    raw_word_list = [word for word in list(raw_word_list) if word not in stop_words]
    return raw_word_list

#step 1:读取文件中的内容组成一个列表
words=word2vec.Text8Corpus(u"./data/corpusSegDone.txt")
# print('Data size', len(words))

begin = time()


model = word2vec.Word2Vec(words, sg=0, size=100,  window=5,  min_count=1,  negative=3, sample=0.001, hs=1, workers=4)
model.save("data/word2vec_gensim")
model.wv.save_word2vec_format("data/word2vec_org",
                              "data/vocabulary",
                              binary=False)

end = time()
print ("Total procesing time: %d seconds" % (end - begin))

# from gensim.models import Word2Vec
# en_wiki_word2vec_model = Word2Vec.load('./data/word2vec_gensim')
# print(en_wiki_word2vec_model.most_similar('家长'))