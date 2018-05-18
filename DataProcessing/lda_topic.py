# -*- coding:utf-8 -*-
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from gensim import corpora, models

#
# train_set = []
# with open('data/qianding_chatlog/result_cut.txt',"r",encoding="UTF-8") as wf:
#     for word in wf:
#         train_set.append(word.split(' '))
#
# # 构建训练语料
# dictionary = Dictionary(train_set)
# corpus = [ dictionary.doc2bow(text) for text in train_set]
#
#
# texts_tf_idf = models.TfidfModel(corpus)[corpus]
# lsi = models.lsimodel.LsiModel(corpus=corpus, id2word=dictionary, num_topics=3)    # 初始化一个LSI转换
# texts_lsi = lsi[texts_tf_idf]                # 对其在向量空间进行转换
# print (lsi.print_topics(num_topics=3, num_words=4))
#
# for doc in texts_lsi:
#     print (doc)
# # lda模型训练
# # lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=100)
# lda = models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=300,update_every=0,passes=20)
# texts_lda = lda[texts_tf_idf]
# print (lda.print_topics(num_topics=3, num_words=4))
# for doc1 in texts_lda:
#     print (doc1)
# lda.print_topics(20)
#
# lda.save('qianding.model')



lda = models.ldamodel.LdaModel.load('qianding.model')
print(lda.print_topics(100))
# 打印id为20的topic的词分布
print(lda.print_topic(100))