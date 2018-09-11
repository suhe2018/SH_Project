import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import cross_validation, metrics
from sklearn.grid_search import GridSearchCV

import matplotlib.pylab as plt

# https://www.cnblogs.com/pinard/p/6143927.html

train = pd.read_csv('./data/train_modified.csv')
target='Disbursed' # Disbursed的值就是二元分类的输出
IDcol = 'ID'
print(train['Disbursed'].value_counts())

x_columns = [x for x in train.columns if x not in [target, IDcol]]
X = train[x_columns]
y = train['Disbursed']

# 默认参数建模
# gbm0 = GradientBoostingClassifier(random_state=10)
# gbm0.fit(X,y)
# y_pred = gbm0.predict(X)
# y_predprob = gbm0.predict_proba(X)[:,1]
# print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
# print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))



#找最佳迭代次数
# param_test1 = {'n_estimators':np.arange(20,81,10)}
# gsearch1 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, min_samples_split=300,
#                                                                min_samples_leaf=20,max_depth=8,max_features='sqrt', subsample=0.8,random_state=10),
#                         param_grid = param_test1, scoring='roc_auc',iid=False,cv=5)
# gsearch1.fit(X,y)
# print(gsearch1.grid_scores_, gsearch1.best_params_, gsearch1.best_score_)



# 找到了一个合适的迭代次数，现在我们开始对决策树进行调参。
# 首先我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索。
# param_test2 = {'max_depth':np.arange(3,14,2), 'min_samples_split':np.arange(100,801,200)}
# gsearch2 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60, min_samples_leaf=20,
#                                                                max_features='sqrt', subsample=0.8, random_state=10),
#                         param_grid = param_test2, scoring='roc_auc',iid=False, cv=5)
# print(gsearch2.fit(X,y))
# print(gsearch2.grid_scores_, gsearch2.best_params_, gsearch2.best_score_)
#输出如下，可见最好的最大树深度是7，内部节点再划分所需最小样本数是300。



# 由于决策树深度7是一个比较合理的值，我们把它定下来，对于内部节点再划分所需最小样本数min_samples_split，
# 我们暂时不能一起定下来，因为这个还和决策树其他的参数存在关联。下面我们再对内部节点再划分所需
# 最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
# param_test3 = {'min_samples_split':np.arange(800,1900,200), 'min_samples_leaf':np.arange(60,101,10)}
# gsearch3 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7,
#                                                                max_features='sqrt', subsample=0.8, random_state=10),
#                         param_grid = param_test3, scoring='roc_auc',iid=False, cv=5)
# gsearch3.fit(X,y)
# gsearch3.grid_scores_, gsearch3.best_params_, gsearch3.best_score_
#输出结果如下，可见这个min_samples_split在边界值，还有进一步调试小于边界60的必要。
# 由于这里只是例子，所以大家可以自己下来用包含小于60的网格搜索来寻找合适的值。


#我们调了这么多参数了，终于可以都放到GBDT类里面去看看效果了。现在我们用新参数拟合数据：
# gbm1 = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60,
#                                   min_samples_split =1200, max_features='sqrt', subsample=0.8, random_state=10)
# gbm1.fit(X,y)
# y_pred = gbm1.predict(X)
# y_predprob = gbm1.predict_proba(X)[:,1]
# print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
# print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))
# 输出如下：
# Accuracy : 0.984
# AUC Score (Train): 0.908099
# 对比我们最开始完全不调参的拟合效果，可见精确度稍有下降，主要原理是我们使用了0.8的子采样，20%的数据没有参与拟合。
# 现在我们再对最大特征数max_features进行网格搜索。

# param_test4 = {'max_features':range(7,20,2)}
# gsearch4 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60,
#                                                                min_samples_split =1200, subsample=0.8, random_state=10),
#                         param_grid = param_test4, scoring='roc_auc',iid=False, cv=5)
# gsearch4.fit(X,y)
# gsearch4.grid_scores_, gsearch4.best_params_, gsearch4.best_score_

# 现在我们再对子采样的比例进行网格搜索：
# param_test5 = {'subsample':[0.6,0.7,0.75,0.8,0.85,0.9]}
# gsearch5 = GridSearchCV(estimator = GradientBoostingClassifier(learning_rate=0.1, n_estimators=60,max_depth=7, min_samples_leaf =60,
#                                                                min_samples_split =1200, max_features=9, random_state=10),
#                         param_grid = param_test5, scoring='roc_auc',iid=False, cv=5)
# gsearch5.fit(X,y)
# gsearch5.grid_scores_, gsearch5.best_params_, gsearch5.best_score_

# 现在我们基本已经得到我们所有调优的参数结果了。这时我们可以减半步长，最大迭代次数加倍来增加我们模型的泛化能力。再次拟合我们的模型：
# gbm2 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=120,max_depth=7, min_samples_leaf =60,
#                                   min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
# gbm2.fit(X,y)
# y_pred = gbm2.predict(X)
# y_predprob = gbm2.predict_proba(X)[:,1]
# print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
# print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

# 输出如下：
# Accuracy : 0.984
# AUC Score (Train): 0.905324
# 可以看到AUC分数比起之前的版本稍有下降，这个原因是我们为了增加模型泛化能力，为防止过拟合而减半步长，最大迭代次数加倍，
# 同时减小了子采样的比例，从而减少了训练集的拟合程度。
# 下面我们继续将步长缩小5倍，最大迭代次数增加5倍，继续拟合我们的模型：
# gbm3 = GradientBoostingClassifier(learning_rate=0.01, n_estimators=600,max_depth=7, min_samples_leaf =60,
#                                   min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
# gbm3.fit(X,y)
# y_pred = gbm3.predict(X)
# y_predprob = gbm3.predict_proba(X)[:,1]
# print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
# print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

# 输出如下，可见减小步长增加迭代次数可以在保证泛化能力的基础上增加一些拟合程度。
# Accuracy : 0.984
# AUC Score (Train): 0.908581
# 最后我们继续步长缩小一半，最大迭代次数增加2倍，拟合我们的模型：
gbm4 = GradientBoostingClassifier(learning_rate=0.005, n_estimators=1200,max_depth=7, min_samples_leaf =60,
                                  min_samples_split =1200, max_features=9, subsample=0.7, random_state=10)
gbm4.fit(X,y)
y_pred = gbm4.predict(X)
y_predprob = gbm4.predict_proba(X)[:,1]
print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, y_pred))
print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, y_predprob))

# 输出如下，此时由于步长实在太小，导致拟合效果反而变差，也就是说，步长不能设置的过小。
# Accuracy : 0.984
# AUC Score (Train): 0.908232

