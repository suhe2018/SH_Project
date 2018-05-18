##!/usr/bin/env python
## coding=utf-8
import jieba

filePath='./data/qianding_chatlog/2018-05-07.txt'
fileSegWordDonePath ='data/qianding_chatlog/result_cut2.txt'
# read the file by line
fileTrainRead = []
#fileTestRead = []
with open(filePath,encoding="UTF-8") as fileTrainRaw:
    for line in fileTrainRaw:
        fileTrainRead.append(line)

stop_words = []
with open('./data/stop_words.txt',"r",encoding="UTF-8") as f:
    line = f.readline()
    while line:
        # line=line.encode("utf-8")
        stop_words.append(line[:-1])
        line = f.readline()
stop_words = set(stop_words)
print('停用词读取完毕，共{n}个词'.format(n=len(stop_words)))
# define this function to print a list with Chinese
def PrintListChinese(list):
    for i in range(len(list)):
        print (list[i])
# segment word with jieba
fileTrainSeg=[]
for i in range(len(fileTrainRead)):
    raw_word_list = [word for word in list(fileTrainRead) if word not in stop_words]
    fileTrainSeg.append([' '.join(list(jieba.cut(fileTrainRead[i][2:],cut_all=False)))])
    if i % 100 == 0 :
        print (i)

# to test the segment result
#PrintListChinese(fileTrainSeg[10])

# save the result
with open(fileSegWordDonePath,'wb') as fW:
    for i in range(len(fileTrainSeg)):
        fW.write(fileTrainSeg[i][0].encode('utf-8'))
        fW.write('\n'.encode('utf-8'))