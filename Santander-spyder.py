# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 17:13:15 2021

Database: https://www.kaggle.com/c/santander-customer-satisfaction
@author: Herikc Brecher
"""

# Import from libraries
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import seaborn as sns
import pickle

import warnings
warnings.filterwarnings("ignore")

# Loading the training dataset in CSV format
training_file = 'data/train.csv'
test_file = 'data/test.csv'
data_training = pd.read_csv(training_file)
test_data = pd.read_csv (test_file)
print(data_training.shape)
print(test_data.shape)

# Viewing the first 20 lines
data_training.head (20)

# Data type of each attribute
data_training.dtypes

# Statistical Summary
data_training.describe()

# Distribution of classes
data_training.groupby("TARGET").size()

# Dividing by class
data_class_0 = data_training[data_training['TARGET'] == 0]
data_class_1 = data_training[data_training['TARGET'] == 1]

counter_class_0 = data_class_0.shape[0]
contador_classe_1 = data_class_1.shape[0]

data_class_0_sample = data_class_0.sample(counter_class_0)
training_data = pd.concat([data_class_0_sample, data_class_1], axis = 0)

# Pearson correlation
data_training.corr(method = 'pearson')

# Finding the correlation between the target variable and the predictor variables
corr = training_data[training_data.columns [1:]].corr()['TARGET'][:].abs()

minimal_correlation = 0.02
corr2 = corr[corr > minimal_correlation]
corr2.shape
corr2

corr_keys = corr2.index.tolist()
data_filter = data_training[corr_keys]
data_filter.head(20)
data_filter.dtypes

# Filtering only the columns that have a correlation above the minimum variable
array_treino = data_training[corr_keys].values

# Separating the array into input and output components for training data
X = array_treino[:, 0:array_treino.shape[1] - 1]
Y = array_treino[:, array_treino.shape[1] - 1]

# Creating the training and test dataset
test_size = 0.30
X_training, X_testing, Y_training, Y_testing = train_test_split(X, Y, test_size = test_size)

# Generating normalized data
scaler = Normalizer (). fit (X_training)
normalizedX_treino = scaler.transform(X_training)

scaler = Normalizer().fit(X_testing)
normalizedX_teste = scaler.transform(X_testing)
Y_training = Y_training.astype('int')
Y_testing = Y_testing.astype('int')

'''
    Execution of a series of classification algorithms is based on those that have the best result.
    For this test, the training base is used without any treatment or data selection.
'''

# Setting the number of folds for cross validation
num_folds = 10

# Preparing the list of models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('NB', GaussianNB()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('SVM', SVC()))

results = []
names = []

for name, model in models:
    kfold = KFold (n_splits = num_folds)
    cv_results = cross_val_score (model, X_training, Y_training, cv = kfold, scoring = 'accuracy')
    results.append (cv_results)
    names.append (name)
    msg = "% s:% f (% f)"% (name, cv_results.mean (), cv_results.std ())
    print (msg)

# Boxplot to compare the algorithms
fig = plt.figure ()
fig.suptitle ('Comparison of Classification Algorithms')
ax = fig.add_subplot (111)
plt.boxplot (results)
ax.set_xticklabels (names)
plt.show ()

# Function to evaluate the performance of the model and save it in a pickle format for future reuse.
def model_report(model_name):
    # Print result
    print("Accuracy:% .3f"% score)
    
    # Making predictions and building the Confusion Matrix
    predictions = result.predict(X_testing)
    matrix = confusion_matrix(Y_testing, predictions)
    print(matrix)
    
    report = classification_report(Y_testing, predictions)
    print(report)
    
    # The precision matrix is ​​created to visualize the number of correct cases
    labels = ['SATISFIED', 'UNSATISFIED']
    cm = confusion_matrix(Y_testing, predictions)
    cm = pd.DataFrame(cm, index = ['0', '1'], columns = ['0', '1'])
     
    plt.figure(figsize = (10.10))
    sns.heatmap(cm, cmap = "Blues", linecolor = 'black', linewidth = 1, annot = True, fmt = '', xticklabels = labels, yticklabels = labels)
    
    # Saving the model
    file = 'models/final_classifier_model' + model_name + '.sav'
    pickle.dump (model, open(file, 'wb'))
    print("Saved Model!")

# Linear Regression
model = LogisticRegression()
result = model.fit(normalizedX_treino, Y_testing)
score = result.score(normalizedX_treino, Y_testing)
model_report("LR")

# Linear Discriminant Analysis
model = LinearDiscriminantAnalysis()
result = model.fit(X_training, Y_testing)
score = result.score(X_training, Y_testing)
model_report("LDA")

# KNN
model = KNeighborsClassifier()
result = model.fit(normalizedX_treino, Y_testing)
score = result.score(normalizedX_treino, Y_testing)
model_report("KNN")

# CART
model = DecisionTreeClassifier()
result = model.fit(X_training, Y_testing)
score = result.score(X_training, Y_testing)
model_report("CART")

# XGBOOST
model = XGBClassifier()
result = model.fit(X_training, Y_testing)
score = result.score(X_training, Y_testing)
model_report("XGBOOST")

# Loading the model
file = 'models model_classifier_final_XGBOOST.sav'
model_classifier = pickle.load(open(file, 'rb'))
model_prod = model_classifier.score(X_testing, Y_testing)
print("Uploaded Model")

# Print Result
print("Accuracy:% .3f"% (model_prod.mean () * 100))











