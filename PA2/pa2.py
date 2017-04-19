import pandas as pd
import numpy as np
from summary import explore
import preprocess as pre 
import classifier 
import sys

# Features
X = np.array(['revolving_utilization_of_unsecured_lines', 'age', 'number_of_time30-59_days_past_due_not_worse',
            'debt_ratio', 'monthly_income','number_of_open_credit_lines_and_loans', 'number_of_times90_days_late',
            'number_real_estate_loans_or_lines','number_of_time60-89_days_past_due_not_worse', 'number_of_dependents'])
# Outcome variable
y = 'serious_dlqin2yrs'

def main(filename):
	df = explore(filename)
	df = pre.impute(df, ['monthly_income'], method = 'mean')
	df = pre.impute(df, ['number_of_dependents'], method ='zero')
	df = pre.discretize(df, 'age', bin_size=5, boundaries=(20,80), method='cut')
	df = pre.discretize(df, 'debt_ratio', bins=4, method='qcut')
	df = pre.cat_to_dummy(df, 'debt_ratio_bucket')
	classifier.evaluate_clf(df, X, y)

if __name__=="__main__":
	if len(sys.argv) != 2:
		print('Input format: python pa2.py <filename path>') 
		sys.exit(1)

	filename = sys.argv[1]
	main(filename)
