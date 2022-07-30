import matplotlib.pyplot as plt
import pandas as pd
import nltk
nltk.download('stopwords')


train_data = pd.read_csv('Data/train.tsv', sep='\t')
test_data = pd.read_csv('Data/test.tsv', sep='\t')

Sentiment_words = []
for row in train_data['Sentiment']:
    if row == 0:
        Sentiment_words.append('negative')
    elif row == 1:
        Sentiment_words.append('somewhat negative')
    elif row == 2:
        Sentiment_words.append('neutral')
    elif row == 3:
        Sentiment_words.append('somewhat positive')
    elif row == 4:
        Sentiment_words.append('positive')
    else:
        Sentiment_words.append('Failed')
train_data['Sentiment_words'] = Sentiment_words


# 缺失值检查
print("Missing value:")
print(train_data.isnull().sum())
# 异常值检查
print("Exception value:")
print(train_data.info())
# Sentiment_words列检查
print("Check column:")
print(train_data.Sentiment_words.unique())
# 查看Sentiment_words列不平衡程度
print("Check value counts:")
BarNum = train_data.Sentiment_words.value_counts()
print(BarNum)
Index = [0, 1, 2, 3, 4]
BarNum.plot.bar()
plt.xticks(Index, ['neutral', 'somewhat positive', 'somewhat negative', 'positive', 'negative'], rotation=45)
plt.xlabel('Sentiment')
plt.ylabel('Frequency')
plt.title('Sample Distribution')
plt.text(0, BarNum[0], BarNum[0], fontsize=10, horizontalalignment='center', verticalalignment='bottom')
plt.text(1, BarNum[1], BarNum[1], fontsize=10, horizontalalignment='center', verticalalignment='bottom')
plt.text(2, BarNum[2], BarNum[2], fontsize=10, horizontalalignment='center', verticalalignment='bottom')
plt.text(3, BarNum[3], BarNum[3], fontsize=10, horizontalalignment='center', verticalalignment='bottom')
plt.text(4, BarNum[4], BarNum[4], fontsize=10, horizontalalignment='center', verticalalignment='bottom')
plt.show()

