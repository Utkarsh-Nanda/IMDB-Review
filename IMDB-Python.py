#!/usr/bin/env python
# coding: utf-8

# Importing Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from matplotlib import style
style.use('ggplot')
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Loading Data
df = pd.read_csv('IMDB Dataset.csv')
df.head()

# Exploratory Data Analysis (EDA)
df.shape
df.info()
sns.countplot(x='sentiment', data=df)
plt.title("Sentiment distribution")

# Displaying first few reviews and their sentiments
for i in range(5):
    print("Review: ", [i])
    print(df['review'].iloc[i], "\n")
    print("Sentiment: ", df['sentiment'].iloc[i], "\n\n")

# Function to count number of words in a review
def no_of_words(text):
    words= text.split()
    word_count = len(words)
    return word_count

# Adding a new column 'word count' to the DataFrame
df['word count'] = df['review'].apply(no_of_words)
df.head()

# Visualizing distribution of word count in reviews
fig, ax = plt.subplots(1,2, figsize=(10,6))
ax[0].hist(df[df['sentiment'] == 'positive']['word count'], label='Positive', color='blue', rwidth=0.9);
ax[0].legend(loc='upper right');
ax[1].hist(df[df['sentiment'] == 'negative']['word count'], label='Negative', color='red', rwidth=0.9);
ax[1].legend(loc='upper right');
fig.suptitle("Number of words in review")
plt.show()

# Data Preprocessing
# Replacing sentiment labels with numeric values
df.sentiment.replace("positive", 1, inplace=True)
df.sentiment.replace("negative", 0, inplace=True)

# Function for text preprocessing
def data_processing(text):
    text = text.lower()
    text = re.sub('<br />', '', text) 
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^\w\s]','', text)
    text_tokens = word_tokenize(text)
    filtered_text = [w for w in text_tokens if not w in stop_words]
    return " ".join(filtered_text)

# Applying data preprocessing to 'review' column
df.review = df['review'].apply(data_processing)

# Handling Duplicates
# Counting and removing duplicate entries
duplicated_count = df.duplicated().sum()
print("Number of duplicate entries: ", duplicated_count)
df = df.drop_duplicates('review')

# Text Stemming
# Stemming words in reviews
stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data
df.review = df['review'].apply(lambda x: stemming(x))

# Adding 'word count' column after text processing
df['word count'] = df['review'].apply(no_of_words)
df.head()

# Word Cloud Visualization
# Creating word cloud for positive reviews
pos_reviews = df[df.sentiment ==1]
text = ' '.join([word for word in pos_reviews['review']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive reviews', fontsize = 19)
plt.show()

# Common Word Analysis for Positive Reviews
from collections import Counter
count = Counter()
for text in pos_reviews['review'].values:
    for word in text.split():
        count[word] +=1
count.most_common(15)
pos_words = pd.DataFrame(count.most_common(15))
pos_words.columns = ['word', 'count']
pos_words.head()
px.bar(pos_words, x='count', y='word', title='Common words in positive reviews', color='word')

# Creating word cloud for negative reviews
neg_reviews = df[df.sentiment == 0]
text = ' '.join([word for word in neg_reviews['review']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative reviews', fontsize = 19)
plt.show()

# Common Word Analysis for Negative Reviews
count = Counter()
for text in neg_reviews['review'].values:
    for word in text.split():
        count[word] += 1
count.most_common(15)
neg_words = pd.DataFrame(count.most_common(15))
neg_words.columns = ['word', 'count']
neg_words.head()
px.bar(neg_words, x='count', y='word', title='Common words in negative reviews', color='word')

# Feature Engineering
# Converting text data into TF-IDF vectors
X = df['review']
Y = df['sentiment']
vect = TfidfVectorizer()
X = vect.fit_transform(df['review'])

# Train-Test Split
# Splitting dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Reducing the size of the dataset for faster training (optional)
x_train = x_train[:2000]
y_train = y_train[:2000]
x_test = x_test[:500]
y_test = y_test[:500]

# Converting sparse matrix to array
x_train = x_train.toarray()
x_test = x_test.toarray()

# Neural Network Model
# Defining a simple feedforward neural network model
model = Sequential()
model.add(Dense(units=16, activation='relu', input_dim=x_train.shape[1]))
model.add(Dense(units=8, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))

# Compiling the model
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Model Training
# Training the model on the training data
history = model.fit(x_train, y_train, batch_size=10, epochs=15)

# Model Summary
model.summary()

# Model Evaluation
# Evaluating model performance on test data
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Test loss:', test_loss)
print('Test accuracy:', test_acc)

# Visualizing Training History
plt.figure(figsize=(16,8))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], color='r', label='loss')
plt.title('Training Loss')
plt.xlabel("Number of epochs")
plt.ylabel("Loss")
plt.subplot(1,2,2)
plt.plot(history)
