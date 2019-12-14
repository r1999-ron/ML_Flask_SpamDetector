# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 02:44:31 2019

@author: KIIT
"""
from flask import Flask,render_template,url_for,request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    df=pd.read_csv('C:\\Users\KIIT\\Downloads\\YouTube-Spam-Collection-v1\\Youtube01-Psy.csv')
    df_data=df[["CONTENT","CLASS"]]
    
    df_x = df_data['CONTENT']
    df_y = df_data.CLASS
    corpus = df_x
    cv = CountVectorizer()
    X = cv.fit_transform(corpus)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train,y_test = train_test_split(X,df_y,test_size=0.33,random_state=42)
    
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB()
    clf.fit(X_train,y_train)
    clf.score(X_test,y_test)
    
   # ytb_model = open("naivebayes_spam_model.pkl","rb")
   # clf = joblib.load(ytb_model)
    
    if request.method == 'POST':
        comment = request.form['comment']
        data = [comment]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('results.html',prediction=my_prediction)







if __name__ == '__main__':
    app.run(debug=True)