import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
import warnings

warnings.filterwarnings('ignore')

# 1 数据准备
# 数据文件是.tsv格式的，也就是说数据用制表符\t分隔，类似于.csv文件的数据用逗号分隔
data_train = pd.read_csv('Data/train.tsv', sep='\t')
data_test = pd.read_csv('Data/test.tsv', sep='\t')
print("(Training Data，Attribute) = ", data_train.shape)
print("(Testing Data，Attribute) = ", data_test.shape)
print()
df1 = data_train.iloc[:, 0:]
print("Original Training Data : No. of negative samples = {}, No. of somewhat negative samples = {}, No. of neutral "
      "samples = {}, No. of somewhat positive samples = {}, No. of positive samples = {}".format(
       df1.loc[df1.iloc[:, -1] == 0, :].shape[0], df1.loc[df1.iloc[:, -1] == 1, :].shape[0],
       df1.loc[df1.iloc[:, -1] == 2, :].shape[0], df1.loc[df1.iloc[:, -1] == 3, :].shape[0],
       df1.loc[df1.iloc[:, -1] == 4, :].shape[0]))
print()

# 提取训练集中的文本内容
train_sentences = data_train['Phrase']
# 提取测试集中的文本内容
test_sentences = data_test['Phrase']
# 通过pandas的concat函数将训练集和测试集的文本内容合并到一起
sentences = pd.concat([train_sentences, test_sentences])
# 提取训练集中的情感标签，一共是156060个标签
label = data_train['Sentiment']
# 导入停词库，停词库中的词是一些废话单词和语气词，对情感分析没什么帮助
stop_words = open('Data/stop_words.txt', encoding='utf-8').read().splitlines()

# 2 文本特征工程 —— 把文本转换成向量 —— 词袋模型
co = CountVectorizer(
    analyzer='word',     # 以词为单位进行分析
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
param_grid_bay = {'alpha': [0.1, 0.5, 1.0]}
skf_bay = StratifiedKFold(n_splits=5, random_state=50, shuffle=True)
bay = MultinomialNB()
grid_bay = GridSearchCV(estimator=bay, param_grid=param_grid_bay, cv=skf_bay)
grid_bay.fit(x_train, y_train)
print(grid_bay.best_params_)
bay_final = grid_bay.best_estimator_
# y_pred_bay = bay_final.predict(x_val)
print("词袋特征提取--朴素贝叶斯分类器验证集上的预测准确率 = {}".format(bay_final.score(x_val, y_val)))

# 3.2 逻辑回归
param_grid_lg = {'C': range(1, 10), 'dual': [True, False]}
skf_lg = StratifiedKFold(n_splits=5, random_state=50, shuffle=True)
lgGS = LogisticRegression()
grid_lg = GridSearchCV(estimator=lgGS, param_grid=param_grid_lg, cv=skf_lg, n_jobs=-1)   # cv=5为5折交叉验证
grid_lg.fit(x_train, y_train)
print(grid_lg.best_params_)
lg_final = grid_lg.best_estimator_
print('词袋特征提取--逻辑回归分类器验证集上的预测准确率 = {}'.format(lg_final.score(x_val, y_val)))