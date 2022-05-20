import jieba
import os
import gensim
import numpy as np
import pickle as pkl
from sklearn.cluster import k_means
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


def add_jieba(words):
    for word in words:
        jieba.add_word(word)


def cluster(data):
    with open('vec_dist', 'rb') as f:
        vec_dist = pkl.load(f)
    vec = []
    for d in data:
        vec.append(vec_dist[d])
    center, label, inertia = k_means(vec, n_clusters=3)
    vec = PCA(n_components=2).fit_transform(vec)

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.scatter(vec[:, 0], vec[:, 1], c=label)
    for i, w in enumerate(data):
        plt.annotate(text=w, xy=(vec[:, 0][i], vec[:, 1][i]),
                     xytext=(vec[:, 0][i] + 0.01, vec[:, 1][i] + 0.01))
    plt.show()


deal_file = False
mode = 'train'
stop_words = open('stop_words.txt').readlines()
kongfu = open('kongfu.txt').readlines()
people = open('people.txt').readlines()
sects = open('sects.txt').readlines()

add_jieba(kongfu)
add_jieba(people)
add_jieba(sects)

delete_char = "\n `1234567890-=/*-~!@#$%^&*()_+qwertyuiop[]\\QWERTYUIOP{}|asdfghjkl;" \
              "'ASDFGHJKL:\"zxcvbnm,./ZXCVBNM<>?~！@#￥%……&*（）——+【】：；“‘’”《》？，。" \
              "、★「」『』～＂□ａｎｔｉ－ｃｌｉｍａｘ＋．／０１２３４５６７８９＜＝＞＠Ａ" \
              "ＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［＼］ｂｄｅｆｇｈｊｋｏｐｒｓ" \
              "ｕｖｗｙｚ￣\u3000\x1a"
file_list = './txt_files/'
data_path = open('data.txt', 'a')
txt_names = os.listdir(file_list)
if deal_file:
    for txt in txt_names:
        if txt == 'inf.txt':
            continue
        file_name = os.path.join(file_list + txt)
        with open(file_name, "r", encoding="gbk", errors="ignore") as file:
            line = file.read()
            line = line.split('。')
            for content in line:
                for char_del in delete_char:
                    content = content.replace(char_del, '')
                content = content.replace('本书来自免费小说下载站更多更新免费电子书请关注', '')
                content = content.replace('更多更新免费电子书请关注www.cr173.com', '')
                if content == '':
                    continue
                content = list(jieba.cut(content))
                temp = ''
                for word in content:
                    temp += word + ' '
                data_path.write(temp + '\n')

if mode == 'train':
    data = gensim.models.word2vec.LineSentence('./data.txt')
    cbow = gensim.models.word2vec.Word2Vec(data, sg=0, vector_size=200, window=5, min_count=5, workers=8)
    cbow.save("./CBOW.model")
    skip_gram = gensim.models.word2vec.Word2Vec(data, sg=1, vector_size=200, window=5, min_count=5, workers=8)
    skip_gram.save("./SKIP_GRAM.model")
else:
    CBOW = gensim.models.word2vec.Word2Vec.load("./CBOW.model")
    SKIP_GRAM = gensim.models.word2vec.Word2Vec.load("./SKIP_GRAM.model")
    character_names = ["黄蓉", "杨过", "张无忌", "令狐冲", "韦小宝", "峨嵋派", "屠龙刀", "蛤蟆功", "葵花宝典"]
    print("Results of CBOW:")
    for tmp_word in character_names:
        print("Related words of {}: ".format(tmp_word),CBOW.wv.most_similar(tmp_word, topn=5))
    print("------------------")
    print("Results of Skip Gram:")
    for tmp_word in character_names:
        print("Related words of {}: ".format(tmp_word), SKIP_GRAM.wv.most_similar(tmp_word, topn=5))


cbow.wv.vectors = cbow.wv.vectors / (np.linalg.norm(cbow.wv.vectors, axis=1).reshape(-1, 1))
vec_dist = dict(zip(cbow.wv.index_to_key, cbow.wv.vectors))
with open('vec_dist', 'wb') as f:
    pkl.dump(vec_dist, f)
cluster(["明教", "昆仑派", "金顶门", "平通镖局", "张无忌", "郭靖", "降龙十八掌", "太祖长拳"])
