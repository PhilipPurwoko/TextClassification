import pandas as pd
import numpy as np
import nltk
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class Data():
    def __init__(self):
        self.train_data = pd.read_csv('train_E6oV3lV.csv')
        self.test_data = pd.read_csv('test_tweets_anuFYb8.csv')
        self.all_data = self.train_data.append(self.test_data,ignore_index=True)
    def __str__(self):
        return f'Combined data shape : {self.all_data.shape} \nTrain data shape : {self.train.shape} \nTest data shape : {self.test.shape}'
    def racist(self):
        return self.all_data[self.all_data['label'] == 1].head()
    def neutral(self):
        return self.all_data[self.all_data['label'] == 0].head()
    
    def remove_pattern(self,text,pattern):
        r = re.findall(pattern,text)
        for i in r:
            text = re.sub(i,'',text)
        return text
    def clean_data(self):
        self.all_data['tidy_tweet'] = np.vectorize(self.remove_pattern)(self.all_data['tweet'],'@[\w]*')
        self.all_data['tidy_tweet'] = self.all_data['tidy_tweet'].str.replace('[^a-zA-Z#]',' ')
        self.all_data['tidy_tweet'] = self.all_data['tidy_tweet'].apply(lambda tweet: ' '.join([word for word in tweet.split() if len(word)>3]))
    def remove_white_space(self):
        self.all_data['tidy_tweet'].apply(lambda tweet: ' '.join(tweet.split()))
    def get_max_len(self):
        self.max_length = self.all_data['tidy_tweet'].str.split(' ').apply(len).max()
    
    def normalize(self):
        tokenize_tweet = self.all_data['tidy_tweet'].apply(lambda tweet:tweet.split())
        stemmer = nltk.stem.porter.PorterStemmer()
        tokenize_tweet = tokenize_tweet.apply(lambda words:[stemmer.stem(word) for word in words])
        
        tokenize_tweet = tokenize_tweet.apply(lambda words:' '.join(words))
        self.all_data['tidy_tweet'] = tokenize_tweet

    def preprocessing(self):
        self.clean_data()
        self.remove_white_space()
        self.normalize()
        self.get_max_len()
        
        return self
    def split(self,data):
        self.train = data[:31962,:]
        self.test_data = data[31962:,:]
        self.x_train,self.x_val,self.y_train,self.y_val = train_test_split(self.train,self.train_data['label'])
        
class Vectorizer():
    def __init__(self,series):
        self.series = series
        self.vectorizer = CountVectorizer()
    def to_vector(self):
        return self.vectorizer.fit_transform(self.series)

class MachineLearning():
    def __init__(self):
        self.model = SVC(gamma='scale',verbose=True)
    def train(self,x_train,y_train):
        self.model.fit(x_train,y_train)
    def predict(self,x_test):
        return self.model.predict(x_test)
    def evaluate(self,y_true,y_pred):
        return classification_report(y_true,y_pred)
    
def main():
    print('Loading Data...')
    data = Data()
    data = data.preprocessing()
    
    print('Preprocessing Data...')
    vectorizer = Vectorizer(data.all_data['tidy_tweet'].tolist())
    vectorized_data = vectorizer.to_vector()

    data.split(vectorized_data)
    
    print('Training Model')
    print('Please wait')
    model = MachineLearning()
    model.train(data.x_train,data.y_train)
    
    print('Evaluating Model...')
    val_pred = model.predict(data.x_val)
    print(model.evaluate(data.y_val,val_pred))
    
    print('Making Prediction...')
    prediction = model.predict(data.test_data)
    print(pd.Series(prediction).value_counts())

    submission = pd.read_csv('test_tweets_anuFYb8.csv')
    submission.drop('tweet',axis=1,inplace=True)
    submission['label'] = prediction

    filename = 'Model-Prediction.csv'
    submission.to_csv(filename,index=False)
    print(f'Prediction exported to "{filename}"')

if __name__=='__main__':
    main()