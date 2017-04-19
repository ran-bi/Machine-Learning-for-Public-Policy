import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import re

# Please create a folder named 'output' to save files and graphs.

def read_data(filename, index=None):
	'''
	Read data into Pandas dataframe. 
	Only support csv file for now. More file types to support later.
	'''

	if 'csv' in filename:
	    if not index:
	    	df = pd.read_csv(filename)
	    else:
	    	df = pd.read_csv(filename, index_col=index)	    
	    df.columns = [camel_to_snake(col) for col in df.columns]
	    return df
	else:
		print ('Please convert file to csv format.')

def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    Source: https://github.com/yhat/DataGotham2013/blob/master/notebooks
    	/3%20-%20Importing%20Data.ipynb
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """

    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def summarize(df, output=True):
	'''
	Generate data summaries, correlation statistics and distribution graphs for given dataframe.

	Inputs:
		df: Pandas dataframe.
		output: boolean. True - save summaries and graphs to output folder.
	'''

	summary = df.describe().transpose()
	for index in summary.index:
		summary.ix[index, 'missing value count'] = df[index].isnull().sum()

	if output:
		summary.to_csv('output/summary.csv')
		df.corr().to_csv('output/correlation.csv')
		print ('Descriptive statistics file saved to output/summary.csv')
		print ('Correlation statistics file saved to output/correlation.csv')
		plot(df)

def plot(df):
	'''
	Plot distirbution for each column. 
	If data type is categorical, plot barchart.
	If data type is numerical, plot histogram.
	'''

	barchart = ['object', 'category']
	for col in df.columns:
		if df[col].dtype.name in barchart:
			dataframe[col].value_counts().plot(kind = 'bar')
			plt.suptitle('Bar Chart - {}'.format(col))
			plt.savefig('output/{}.png'.format(col))
			print('Graph saved as {}.png'.format(col))
			plt.close()
		else:
			df[col].hist()
			plt.suptitle('Histogram - {}'.format(col))
			plt.savefig('output/{}.png'.format(col))
			print('Graph saved as {}.png'.format(col))
			plt.close()

def explore(filename, index=None, output=True):
	'''
	Main function to implement 1. Read Data; 2. Explore Data.
	'''
	df = read_data(filename, index)
	summarize(df, output)
	return df



