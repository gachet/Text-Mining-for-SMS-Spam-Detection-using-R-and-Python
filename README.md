# Text-Mining-for-SMS-Spam-Detection

Mining for SMS Spam detection using Machine Learning techniques

This folder contain algorithm for SMS - Spam detection. 

You can download dataset from -- https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection or Dataset is provided in this folder itself.

// Algorithm is written in both languages - R and Python.

This problem comes under classification problem. 
To solve this problem, there are many algorithms, but I specifically concentrated on Naive Bayes and SVM.

-- First you need to convert message text into tokens
-- And then apply Naive Bayes algorithm

Following Libraries used while developing algorithm in Python

%matplotlib inline

import matplotlib.pyplot as plt

import csv

from textblob import TextBlob.

import pandas

import sklearn

import cPickle

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from sklearn.naive_bayes import MultinomialNB

from sklearn.svm import SVC, LinearSVC

from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

from sklearn.pipeline import Pipeline

from sklearn.grid_search import GridSearchCV

from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split 

from sklearn.tree import DecisionTreeClassifier 

from sklearn.learning_curve import learning_curve

Note -- To run magic command in python, you need iPython Notebook. These magic commands will not run on normal python. 
It will give syntax error.


Following Libraries used while developing algorithm in R:

library(NLP)

library(tm)         ## for text mining

library(SnowballC)  ## provides wordstem() function

library(wordcloud)  ## for visualizing text data in the form of cloud

library(e1071)      ## for naive bayes implementation

library(gmodels)


