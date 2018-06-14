# coding:utf-8

import sys
import gensim
import sklearn
import numpy as np
from gensim.models import word2vec
from gensim.models.word2vec import LineSentence
from gensim.models.doc2vec import Doc2Vec, LabeledSentence
from time import time

TaggededDocument = gensim.models.doc2vec.TaggedDocument

def get_datasest():
    with open("/Users/suhe/Downloads/QA-master/data/400_cut.txt", 'r',encoding="UTF-8") as cf:
        docs = cf.readlines()
        print (len(docs))

    x_train = []
    #y = np.concatenate(np.ones(len(docs)))
    for i, text in enumerate(docs):
        word_list = text.split(' ')
        l = len(word_list)
        word_list[l-1] = word_list[l-1].strip()
        document = TaggededDocument(word_list, tags=[i])
        x_train.append(document)

    return x_train

def getVecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]].reshape(1, size)) for z in corpus]
    return np.concatenate(vecs)

def train(x_train, size=200, epoch_num=1):
    model_dm = Doc2Vec(x_train,min_count=1, window = 3, size = size, sample=1e-3, negative=5, workers=4)
    model_dm.train(x_train, total_examples=model_dm.corpus_count, epochs=70)
    model_dm.save('./model_dm_wangyi')

    return model_dm

def test():
    model_dm = Doc2Vec.load("./model_dm_wangyi")
    test_text = ['坏了']
    inferred_vector_dm = model_dm.infer_vector(test_text)
    # print (inferred_vector_dm)
    sims = model_dm.docvecs.most_similar([inferred_vector_dm], topn=10)


    return sims

def word2vecter():
    #step 1:读取文件中的内容组成一个列表
    words=word2vec.Text8Corpus(u"/Users/suhe/Downloads/QA-master/data/400_cut.txt")
    # print('Data size', len(words))

    begin = time()


    model = word2vec.Word2Vec(words, sg=0, size=100,  window=5,  min_count=1,  negative=3, sample=0.001, hs=1, workers=4)
    model.save("data/word2vec_gensim")
    model.wv.save_word2vec_format("data/word2vec_org",
                                  "data/vocabulary",
                                  binary=False)

    end = time()
    print ("Total procesing time: %d seconds" % (end - begin))

def testVec():
    from gensim.models import Word2Vec
    model = Word2Vec.load('./data/word2vec_gensim')

    res=model.most_similar('门岗')
    for word in res:
        print(word[0])

    # try:
    #     c = model['boom']
    # except KeyError:
    #     print ("not in vocabulary")
    #     c = 0

    print(type(res))

if __name__ == '__main__':
    # x_train = get_datasest()
    # # model_dm = train(x_train)
    #
    # sims = test()
    # for count, sim in sims:
    #     sentence = x_train[count]
    #     words = ''
    #     for word in sentence[0]:
    #         words = words + word + ' '
    #     print (words, sim, len(sentence[0]) )
    # word2vecter()
    # testVec()
    import os
    f=open('./data/res.txt','w',encoding='utf-8')
    for line in open('./data/tongxing.txt','r',encoding='utf8'):
        words=line.split(',')
        for word in words:
            if(words[0]=='中介看房'):
                f.writelines(words[0]+','+word+'\n')

