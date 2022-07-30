import re
import nltk
from nltk import word_tokenize
from nltk import WordNetLemmatizer
from nltk.corpus import stopwords
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

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


def review_to_words(raw_review):
    review = raw_review
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    lemmatizer = WordNetLemmatizer()
    review = [lemmatizer.lemmatize(w) for w in review if not w in set(stopwords.words('english'))]
    return(' '.join(review))


corpus = []
for i in range(0, 156060):
    corpus.append(review_to_words(train_data['Phrase'][i]))
corpus1 = []
for i in range(0, 156060):
    corpus1.append(review_to_words(train_data['Phrase'][i]))

#positive
train_data['new_Phrase'] = corpus
train_data.drop(['Phrase'], axis=1, inplace=True)
positive = train_data[train_data['Sentiment_words'] == 'positive']
words = ' '.join(positive['new_Phrase'])
s = word_tokenize(words)
s1 = nltk.pos_tag(s)
adj_words1 = ' '.join([name for name, value in s1 if value in ['JJ','JJR', 'JJS','UH']])
split_word = " ".join([word for word in adj_words1.split()])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500,
                      collocations=False  # point
                      ).generate(split_word)
plt.figure(1, figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#somewhat positive
somewhat_positive = train_data[train_data['Sentiment_words'] == 'somewhat positive']
words = ' '.join(somewhat_positive['new_Phrase'])
s = word_tokenize(words)
s1 = nltk.pos_tag(s)
adj_words2 = ' '.join([name for name, value in s1 if value in ['JJ', 'JJR', 'JJS', 'UH']])

split_word = " ".join([word for word in adj_words2.split()])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500,
                      collocations=False
                      ).generate(split_word)
plt.figure(1, figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#neutral
neutral = train_data[train_data['Sentiment_words'] == 'neutral']
words = ' '.join(neutral['new_Phrase'])
s = word_tokenize(words)
s1 = nltk.pos_tag(s)
adj_words3 = ' '.join([name for name, value in s1 if value in ['JJ', 'JJR', 'JJS', 'UH']])

split_word = " ".join([word for word in adj_words3.split()])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500,
                      collocations=False
                      ).generate(split_word)
plt.figure(1, figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#negative
negative = train_data[train_data['Sentiment_words'] == 'negative']
words = ' '.join(negative['new_Phrase'])
s = word_tokenize(words)
s1 = nltk.pos_tag(s)
adj_words4 = ' '.join([name for name, value in s1 if value in ['JJ', 'JJR', 'JJS', 'UH']])

split_word = " ".join([word for word in adj_words4.split()])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500,
                      collocations=False
                      ).generate(split_word)
plt.figure(1, figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

#somewhat negative
somewhat_negative = train_data[train_data['Sentiment_words'] == 'somewhat negative']
words = ' '.join(somewhat_negative['new_Phrase'])
s = word_tokenize(words)
s1 = nltk.pos_tag(s)
adj_words5 = ' '.join([name for name, value in s1 if value in ['JJ', 'JJR', 'JJS', 'UH']])
split_word = " ".join([word for word in adj_words5.split()])
wordcloud = WordCloud(stopwords=STOPWORDS,
                      background_color='black',
                      width=3000,
                      height=2500,
                      collocations=False
                      ).generate(split_word)
plt.figure(1, figsize=(13, 13))
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

