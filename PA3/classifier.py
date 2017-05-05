### ML pipeline 3/3: Build, select and evaluate classifier
### Name: Ran Bi
### Below code is modified from https://github.com/rayidghani/magicloops/blob/master/magicloops.py


from __future__ import division
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid, GridSearchCV
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import seaborn as sns
import sys

# Please create a folder named 'evaluation' to save files and graphs.

def define_clfs_params(grid_type = 'standard'):
    '''
    Get classifiers and hyperparameters grid to test.

    Input:
        grid_type: string. 'standard' or 'test'

    Output:
        a dictionary of classifiers and a dictionary of parameters
    '''

    clfs = {'RF': RandomForestClassifier(n_estimators=50, n_jobs=-1),        
			'LR': LogisticRegression(penalty='l1', C=1e5),
			'SVM': svm.SVC(kernel='linear', probability=True, random_state=0),
			'DT': DecisionTreeClassifier(),
			'KNN': KNeighborsClassifier(n_neighbors=3),
            'NB': GaussianNB(),
			'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), algorithm="SAMME", n_estimators=200),
			'BAG': BaggingClassifier(DecisionTreeClassifier(max_depth=10), n_estimators=5, max_samples=0.65, max_features=1)
			    }

    standard_grid = { 
	    	'RF': {'n_estimators': [1,10,100], 'max_depth': [1,5,10,20,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
		    'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
		    'SVM':{'C' :[0.1,1,10],'kernel':['linear']},
		    'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
			'KNN':{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},	
		    'NB' : {},
            'AB': { 'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100, 1000,10000]},
		    'BAG': {'n_estimators': [5,10,20], 'max_samples':[0.35,0.5,0.65]}
	    		}

    test_grid = { 
		    'RF':{'n_estimators': [1], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},
		    'LR': { 'penalty': ['l1'], 'C': [0.01]},
		    'SVM' :{'C' :[0.01],'kernel':['linear']},
		    'DT': {'criterion': ['gini'], 'max_depth': [1], 'max_features': ['sqrt'],'min_samples_split': [10]},	    
		    'KNN' :{'n_neighbors': [5],'weights': ['uniform'],'algorithm': ['auto']},
		    'AB': { 'algorithm': ['SAMME'], 'n_estimators': [1]},
		    'BAG':{'n_estimators': [5]}
		        }
    if grid_type == 'test':
    	return clfs, test_grid
    else:
    	return clfs, standard_grid


def clf_loop(models_to_run, clfs, grid, X, y, test_size=0.25, output=True):
    '''
    Loop over classifier-parameter combinations and evaluate each classifier by several metrics.

    Inputs:
        models_to_run: list of string. a list of classifiers to test
        clfs: dictionary of classifiers (output of define_clfs_params)
        grid: dictionary of parameters grid (output of define_clfs_params)
        X: a Pandas dataframe of features
        y: a Pandas dataframe of the label
        test_size: float between 0.0 and 1.0 representing the proportion of the dataset to include in the test split. default to 0.25
        output: bool. True - save the output dataframe to csv file

    Output:
        a Pandas dataframe including classifiers, parameters, runtime, and evaluation score
    '''

    results_df =  pd.DataFrame(columns=('model_type','clf', 'parameters', 'train_time', 'test_time',
    									'accuracy','f1_score', 'precision', 'recall', 'auc', 
    									'p_at_5', 'p_at_10', 'p_at_20',
    									'r_at_5', 'r_at_10', 'r_at_20'))
    for n in range(1, 2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state=0)
        for index,clf in enumerate([clfs[x] for x in models_to_run]):
            print (models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    print (clf)
                    train_start = time.time()
                    clf.fit(X_train, y_train)
                    train_end = time.time()
                    train_time = train_end - train_start

                    test_start = time.time()
                    y_pred = clf.predict(X_test)
                    test_end = time.time()
                    test_time = test_end - test_start
                    print('prediction done')

                    y_pred_probs = clf.predict_proba(X_test)[:,1]

                    print('prediction_prob done')
                    scores = evaluate(y_test, y_pred, y_pred_probs)
                    print('eva done')
                    current_count = len(results_df)
                    model_name = models_to_run[index] + str(current_count)
                    results_df.loc[current_count] = [models_to_run[index],clf, p, train_time, test_time,
                                                       scores['accuracy'], scores['f1_score'], scores['precision'],
                                                       scores['recall'], scores['auc'], 
                                                       scores['p_at_5'], scores['p_at_10'], scores['p_at_20'],
                                                       scores['r_at_5'], scores['r_at_10'], scores['r_at_20']]
                    if output:
                        plot_precision_recall_n(y_test, y_pred_probs,model_name)
                except IndexError as e:
                    print ('Error:',e)
                    continue
    if output:
    	results_df.to_csv('evaluation/clf_evaluations.csv')
    return results_df

def evaluate (y_true, y_pred, y_pred_probs):
    '''
    Given a classifier, evaluate by various metrics

    Input:
        y_true: a Pandas dataframe of actual label value
        y_pred: a Pandas dataframe of predicted label value
        y_pred_probs: a Pandas dataframe of probability estimates

    Output:
        rv: a dictionary where key is the metric and value is the score

    '''
	rv = {}

	metrics = {'accuracy': accuracy_score, 'f1_score': f1_score,
	            'precision': precision_score, 'recall': recall_score,
	            'auc': roc_auc_score}
	for metric, fn in metrics.items():
		rv[metric] = fn(y_true, y_pred)

	y_pred_probs_sorted, y_true_sorted = zip(*sorted(zip(y_pred_probs, y_true), reverse=True))
	levels = [5, 10, 20]
	for k in levels:
		rv['p_at_'+str(k)] = precision_at_k(y_true_sorted, y_pred_probs_sorted, k)
		rv['r_at_'+str(k)] = recall_at_k(y_true_sorted, y_pred_probs_sorted, k)

	return rv

def generate_binary_at_k(y_pred_probs, k):
    '''
    Transform probability estimates into binary at threshold of k
    '''

    cutoff_index = int(len(y_pred_probs) * (k / 100.0))
    y_pred_binary = [1 if x < cutoff_index else 0 for x in range(len(y_pred_probs))]
    return y_pred_binary

def precision_at_k(y_true, y_pred_probs, k):
    '''
    Calculate precision score for probability estimates at threshold of k
    '''

    preds_at_k = generate_binary_at_k(y_pred_probs, k)
    precision = precision_score(y_true, preds_at_k)
    return precision

def recall_at_k(y_true, y_pred_probs, k):
    '''
    Calculate recall score for probability estimates at threshold of k
    '''

	preds_at_k = generate_binary_at_k(y_pred_probs, k)
	recall = recall_score(y_true, preds_at_k)
	return recall

def plot_precision_recall_n(y_true, y_pred_probs, model_name):
    '''
    '''

    from sklearn.metrics import precision_recall_curve
    y_score = y_pred_probs
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    plt.figure()
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')

    plt.title(model_name)
    plt.savefig('evaluation/'+model_name)
    plt.close()

# Not used in PA3
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
	df_pred.to_csv('evaluation/prediction.csv')
	return df_pred



