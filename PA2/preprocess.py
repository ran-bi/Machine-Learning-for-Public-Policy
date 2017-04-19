import pandas as pd
import numpy as np
import csv

# Please create a folder named 'output' to save files and graphs.

def impute(df, cols, method = 'mean', output = False):
	'''
	Fill in missing values.

	Inputs:
		df: Pandas dataframe
		cols: list of column names (string) to perform imputation.
		method: 
			'mean' - use mean to fill in missing values 
			'zero' - use 0 to fill in missing values 
			more imputation methods to support in future
		output: boolean. True - save the dataset after imputation as csv file.
	'''

	if method == 'mean':
		df = fill_by_mean(df, cols)
	elif method == 'zero':
		df = fill_by_zero(df, cols)

	if output:
		df.to_csv('output/filled_missing.csv')
		print ('File saved to output/filled_missing.csv')
	return df

def fill_by_mean(df, cols):
	'''
	Fill in missing values by mean.

	Inputs:
		df: Pandas dataframe
		cols: list of column names (string) to impute by mean.
	'''
	for col in cols:
		mean = df[col].mean()
		df[col] = df[col].fillna(mean)
		print ('Filling missing value for {}'.format(col))
	return df

def fill_by_zero(df, cols):
	'''
	Fill in missing values by 0.

	Inputs:
		df: Pandas dataframe
		cols: list of column names (string) to impute by zero.		
	'''
	for col in cols:
		df[col] = df[col].fillna(0)
		print ('Filling missing value for {}'.format(col))
	return df

def discretize(df, col, bins=5, bin_size=None, boundaries=None, method='cut'):
	'''
	Discretize a continuous variable into a new column. It supports 3 methods of discretization:
	1. Quantile-based discretization. Discretize variable into equal-sized buckets based on rank or 
	   based on sample quantiles;
	2. Bin column into buckets of equal sizes. User indicates number of bins.
	3. Bin column into buckets of different sizes. Such method creates larger bins near the max/min
	   values to account for outliers, and equal size bins between certain range. User indicates the
	   size of equal size bins, and boundaries of equal size bins range. 
	   (Inspired by https://github.com/yhat/DataGotham2013/blob/master/notebooks/7%20-%20Feature%20Engineering.ipynb)

	Inputs:
		df: Pandas dataframe.
		col: string. name of the column to bin.
		bins: int. number of bins (for method 1 and 2 only)
		bin_size: int. size of equal size buckets. (for method 3 only)
		boundaries: tuple. range of values to bin into equal size buckets. (for method 3 only)
		method: 
			'cut': method 2 or 3
			'qcut': method 1

	Output:
		dataframe with new column

	'''
	assert df[col].dtype != object
	bucket_col = col + '_bucket'

	if method == 'qcut':		
		df[bucket_col] = pd.qcut(df[col], bins)
		print ('Column {} is discretized in to {} buckets based on quantile'.format(col, str(bins)))

	elif method == 'cut':
		if bin_size:
			min_val = min(df[col].values)
			max_val = max(df[col].values)

			if boundaries:
				lb, ub = boundaries
				assert lb >= min_val
				assert ub <= max_val
				bins = [min_val] + list(range(lb, ub, bin_size)) + [ub, max_val]
			else:
				right = min_val
				bins = [min_val]
				while right < u_b:
					right += bin_size
					bins.append(right)

			df[bucket_col] = pd.cut(df[col], bins, include_lowest=True)
			print ('Column {} is discretized in to {} buckets'.format(col, str(len(bins)-1)))

		else:
			df[bucket_col] = pd.cut(df[col], bins, include_lowest=True)
			print ('Column {} is discretized in to {} buckets'.format(col, str(bins)))

	return df


def cat_to_dummy(df, col):
	'''
	Convert categorical variable into new columns of dummy/indicator variables

	Inputs:
		df: Pandas dataframe
		col: string. name of the column to create dummy

	Output:
		dataframe with new dummy variable columns
	'''
	dummy_col = pd.get_dummies(df[col])
	print ('Column {} is converted in to dummy variables.'.format(col))
	return pd.concat([df, dummy_col], axis=1)








