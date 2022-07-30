import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# 1 数据准备
# 数据文件是.tsv格式的，也就是说数据用制表符\t分隔，类似于.csv文件的数据用逗号分隔
data_train = pd.read_csv('Data/train.tsv', sep='\t')
data_test = pd.read_csv('Data/test.tsv', sep='\t')
print("(训练集，属性数) = ", data_train.shape)
print("(测试集，属性数) = ", data_test.shape)
df1 = data_train.iloc[:, 0:]
print("Original Training Data : No. of negative samples = {}, No. of somewhat negative samples = {}, No. of neutral "
      "samples = {}, No. of somewhat positive samples = {}, No. of positive samples = {}".format(
       df1.loc[df1.iloc[:, -1] == 0, :].shape[0], df1.loc[df1.iloc[:, -1] == 1, :].shape[0],
       df1.loc[df1.iloc[:, -1] == 2, :].shape[0], df1.loc[df1.iloc[:, -1] == 3, :].shape[0],
       df1.loc[df1.iloc[:, -1] == 4, :].shape[0]))

# 欠采样处理
dt0 = data_train[df1.iloc[:, -1] == 0]   # negative
dt1 = data_train[df1.iloc[:, -1] == 1]   # somewhat negative
dt2 = data_train[df1.iloc[:, -1] == 2]   # neutral
dt3 = data_train[df1.iloc[:, -1] == 3]   # somewhat positive
dt4 = data_train[df1.iloc[:, -1] == 4]   # positive
dt1_sample = dt1.sample(n=7072)
dt2_sample = dt2.sample(n=7072)
dt3_sample = dt3.sample(n=7072)
dt4_sample = dt4.sample(n=7072)
new_train_dataset = pd.concat([dt0, dt1_sample, dt2_sample, dt3_sample, dt4_sample], axis=0)

# 提取训练集中的文本内容
train_sentences = new_train_dataset['Phrase']
# 提取测试集中的文本内容
test_sentences = data_test['Phrase']
# 通过pandas的concat函数将训练集和测试集的文本内容合并到一起
sentences = pd.concat([train_sentences, test_sentences])
# 提取训练集中的情感标签，一共是156060个标签
label = new_train_dataset['Sentiment']
# 导入停词库，停词库中的词是一些废话单词和语气词，对情感分析没什么帮助
stop_words = open('Data/stop_words.txt', encoding='utf-8').read().splitlines()

# 2 文本特征工程 —— 把文本转换成向量 —— 词袋模型
co = CountVectorizer(
    analyzer='word',  # 以词为单位进行分析
    ngram_range=(1, 4),  # 分析相邻的几个词，避免原始的词袋模型中词序丢失的问题
    stop_words=stop_words,
    max_features=150000  # 指最终的词袋矩阵里面包含语料库中出现次数最多的多少个词
)
co.fit(sentences)
# 将训练集随机拆分为新的训练集和验证集
x_train, x_val, y_train, y_val = train_test_split(train_sentences, label, test_size=0.30, random_state=50)
# 把训练集和验证集中的每一个词都进行特征工程，变成向量
x_train = co.transform(x_train)
x_val = co.transform(x_val)

# 3 构建分类器算法，对词袋模型处理后的文本进行机器学习和数据挖掘
# 3.1 朴素贝叶斯
classifier = MultinomialNB()
classifier.fit(x_train, y_train)
print('词袋方法进行文本特征工程，使用sklearn默认的多项式朴素贝叶斯分类器，验证集上的预测准确率:', classifier.score(x_val, y_val))

# 3.2 逻辑回归
lg1 = LogisticRegression()
lg1.fit(x_train, y_train)
print('词袋方法进行文本特征工程，使用sklearn默认的逻辑回归分类器，验证集上的预测准确率:', lg1.score(x_val, y_val))
