import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import sys

# Please create a folder named 'output' to save files and graphs.

# Add classifiers user intend to build into the list below and import module from sklearn accordingly.
classifiers = [LogisticRegression(), KNeighborsClassifier(), DecisionTreeClassifier()]

def evaluate_clf(df, X, y, test_size=0.25):
	'''
	Split data into training and testing sets.
	Build classifiers indicated by the user, evaluate each classifier based on accuracy / 
	recall / precision matrics and generate evaluation report into a txt file.

	Inputs:
		df: Pandas dataframe.
		X: array of string. features.
		y: string. output variable.
		test_size: proportion of the data to be used as test dataset.

	Output:
		Evaluation report in output folder.
	'''

	X_train, X_test, y_train, y_test  = train_test_split(df[X], df[y], test_size=0.25)
	print('Building classifiers...')

	with open ('output/clf_evaluation.txt', 'w') as f:
		for clf in classifiers:
			clf.fit(X_train, y_train)
			y_pred = clf.predict(X_test)
			accuracy = metrics.accuracy_score(y_test, y_pred)
			recall = metrics.recall_score(y_test, y_pred)
			precision = metrics.precision_score(y_test, y_pred)
			f.write('{} classifier built.\n'.format(clf))
			f.write('Accuracy score is: {}\n'.format(accuracy))
			f.write('Recall score is: {}\n'.format(recall))
			f.write('Precision score is: {}\n'.format(precision))
			f.write('='*77 + '\n')
	f.close()
	print('Classifiers evaluation report saved to output/clf_evaluation.txt')


def predict(df_train, df_pred, X, y, clf=LogisticRegression(), output=True):
	'''
	Given features data, predict value based on the classifier user chooses, and add 
	predicted values to the dataset.
	
	Inputs:
		df_train: Pandas dataframe. complete dataset with both features and output variable 
			to train the model.
		df_pred: Pandas dataframe. dataset with features only.
		X: array of string. features.
		y: string. output variable.
		clf: classifier to use.
		output: boolean. True: save the outcome dataframe into csv file.

	Output:
		Pandas dataframe. Original df_pred with predicted value of predicted variable.
	'''
	clf.fit(df_train[X], df_train[y])
	y_pred = clf.predict(df_pred[X])
	df_pred[y] = y_pred
	df_pred.to_csv('output/prediction.csv')
	return df_pred



