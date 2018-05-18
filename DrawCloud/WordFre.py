# -*- coding:utf-8 -*-
import pandas as pd
word_lst = []
word_dict= {}
with open('data/qianding_chatlog/result_cut.txt',"r",encoding="UTF-8") as wf:

    for word in wf:
        word_lst.append(word.split(' '))
        for item in word_lst:
            for item2 in item:
                if item2 not in word_dict:
                    word_dict[item2] = 1
            else:
                word_dict[item2] += 1
a=[]
b=[]
for key in word_dict:
    print (key,word_dict[key])
    a.append(key)
    b .append( word_dict[key])

    #字典中的key值即为csv中列名
dataframe = pd.DataFrame({'a_name':a,'b_name':b})

    #将DataFrame存储为csv,index表示是否显示行名，default=True
dataframe.to_csv("data/qianding_chatlog/test.csv",index=False,sep=',')
