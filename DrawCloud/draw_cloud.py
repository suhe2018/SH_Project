import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from scipy.misc import imread
from random import choice
import pandas as pd

# 定义颜色，方法很多，这里用到的方法是在四个颜色中随机抽取
def my_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    return choice(["rgb(94,38,18)", "rgb(41,36,33)", "rgb(128,128,105)", "rgb(112,128,105)"])

def draw_cloud(mask_path, word_freq, save_path):
    mask = imread(mask_path)  #读取图片
    wc = WordCloud(font_path='data/qianding_chatlog/kaiti.TTF',  # 设置字体
                   background_color="white",  # 背景颜色
                   max_words=500,  # 词云显示的最大词数
                   mask=mask,  # 设置背景图片
                   max_font_size=80,  # 字体最大值
                   random_state=42,
                   )
    # generate_from_frequencies方法，从词频产生词云输入
    wc.generate_from_frequencies(word_freq)

    plt.figure()

    # 刘峰， 采用自定义颜色
    plt.imshow(wc.recolor(color_func=my_color_func), interpolation='bilinear')

    # 何小嫚， 采用图片底色
    # image_colors = ImageColorGenerator(mask)
    # plt.imshow(wc.recolor(color_func=image_colors), interpolation='bilinear')

    plt.axis("off")
    wc.to_file(save_path)
    plt.show()

# 获取关键词及词频
# input_freq = person_word("刘峰")
# 经过手动调整过的词频文件,供参考
freq = pd.read_csv("data/qianding_chatlog/test.csv", header=None, index_col=0, skipinitialspace=True, skiprows=1)
input_freq = freq[1].to_dict()
draw_cloud("data/qianding_chatlog/he.png", input_freq, "data/output/he.png")
