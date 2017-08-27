# -*- coding:utf-8 -*-

import numpy as np
from gensim import corpora, models, similarities
import time
import matplotlib.pyplot as plt
from pylab import *


def load_stopword():
    f_stop = open('sw.txt')
    sw = [line.strip() for line in f_stop]
    f_stop.close()
    return sw

# remove the stop words
sw = load_stopword()
f = open('news_cn.dat', encoding='utf-8')    # load the text data
texts = [[word for word in line.strip().lower().split() if word not in sw] for line in f]
f.close()
M = len(texts)
print('语料库载入完成，据统计，一共有 %d 篇文档' % M)

# build the dictionary for texts
dictionary = corpora.Dictionary(texts)
dict_len = len(dictionary)
# transform the whole texts to sparse vector
corpus = [dictionary.doc2bow(text) for text in texts]
# create a transformation, from initial model to tf-idf model
corpus_tfidf = models.TfidfModel(corpus)[corpus]
print('现在开始训练LDA模型：---')
num_topics = 9
t_start = time.time()
# create a transformation, from tf-idf model to lda model
lda = models.LdaModel(corpus_tfidf, num_topics=num_topics, id2word=dictionary,
      alpha=0.01, eta=0.01, minimum_probability=0.001, update_every = 1, chunksize = 100, passes = 1)
print('LDA模型完成，耗时 %.3f 秒' % (time.time() - t_start))

# 打印前9个文档的主题
num_show_topic = 9  # 每个文档显示前几个主题
print('下面，显示前9个文档的主题分布：')
doc_topics = lda.get_document_topics(corpus_tfidf)  # 所有文档的主题分布
for i in range(9):
    topic = np.array(doc_topics[i])
    topic_distribute = np.array(topic[:, 1])
    topic_idx = list(topic_distribute)
    print('第%d个文档的 %d 个主题分布概率分别为：' % (i, num_show_topic))
    print(topic_idx)

print('\n下面，显示每个主题的词分布：')
num_show_term = 7   # 每个主题下显示几个词
for topic_id in range(num_topics):
    print('第%d个主题的词与概率如下：\t' % topic_id)
    term_distribute_all = lda.get_topic_terms(topicid=topic_id)
    term_distribute = term_distribute_all[:num_show_term]
    term_distribute = np.array(term_distribute)
    term_id = term_distribute[:, 0].astype(np.int)
    print('词：\t', end='  ')
    for t in term_id:
        print(dictionary.id2token[t], end=' ')
    print('\n概率：\t', term_distribute[:, 1])

# 下面，画出9个主题的主要的7个词的概率分布图
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
for i, k in enumerate(range(num_topics)):
    ax = plt.subplot(3, 3, i+1)
    item_dis_all = lda.get_topic_terms(topicid=k)
    item_dis = np.array(item_dis_all[:num_show_term])
    ax.plot(range(num_show_term), item_dis[:, 1], 'b*')
    item_word_id = item_dis[:, 0].astype(np.int)
    word = [dictionary.id2token[i] for i in item_word_id]
    ax.set_ylabel(u"概率")
    for j in range(num_show_term):
        ax.text(j, item_dis[j, 1], word[j], bbox=dict(facecolor='green',alpha=0.1))
plt.suptitle(u'9个主题及其7个主要词的概率', fontsize=18)
plt.show()

# 下面，画出前9个文档分别属于9个主题的概率
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    doc_item = np.array(doc_topics[i])
    doc_item_id = np.array(doc_item[:, 0])
    doc_item_dis = np.array(doc_item[:, 1])
    ax.plot(doc_item_id, doc_item_dis, 'r*')
    for j in range(doc_item.shape[0]):
        ax.text(doc_item_id[j], doc_item_dis[j], '%.3f' % doc_item_dis[j])
plt.suptitle(u'前9篇文档的主题分布图', fontsize=18)
plt.show()



