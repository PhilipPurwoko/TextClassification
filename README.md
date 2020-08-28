# TextClassification
Implementasi Python dengan deep learning menggunakan tensorflow untuk Sentiment Analysis

# Projects
## IMDB - Classification
Klasifikasi komentar di IMDB dengan Deep Neural Network. NN Model dilatih menggunakan keras dataset. IMDB movie review sentiment classification dataset (https://keras.io/api/datasets/imdb/). This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing (https://ai.stanford.edu/~amaas/data/sentiment/)

**Dataset**
Total terdapat 50000 review terdiri dari 25000 training data dan 25000 test data. Merupakan binary classification atau klasifikasi untuk 2 kelas. Yakni 0 untuk review negatif dan 1 untuk review positif

## Twitter - Classification
Klasifikasi tweet dengan Deep Neural Network. NN Model dilatih dengan dataset berupa tweet dan label. Label menyatakan apakah tweet tersebut mengandung unsur racist atau tidak. The objective of this task is to detect hate speech in tweets. For the sake of simplicity, we say a tweet contains hate speech if it has a racist or sexist sentiment associated with it. So, the task is to classify racist or sexist tweets from other tweets.

Formally, given a training sample of tweets and labels, where label '1' denotes the tweet is racist/sexist and label '0' denotes the tweet is not racist/sexist, your objective is to predict the labels on the test dataset.

**Dataset**
Our overall collection of tweets was split in the ratio of 65:35 into training and testing data. Out of the testing data, 30% is public and the rest is private (https://datahack.analyticsvidhya.com/contest/practice-problem-twitter-sentiment-analysis/)
1. train.csv - For training the models, we provide a labelled dataset of 31,962 tweets. The dataset is provided in the form of a csv file with each line storing a tweet id, its label and the tweet.
There is 1 test file (public)
2. test_tweets.csv - The test data file contains only tweet ids and the tweet text with each tweet in a new line