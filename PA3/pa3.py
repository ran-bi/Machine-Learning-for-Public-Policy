import pandas as pd
import numpy as np
from summary import explore
import preprocess as pre 
import classifier as cl
import sys


# Configuration
X = np.array(['revolving_utilization_of_unsecured_lines', 'debt_ratio', 'age', 'monthly_income', 'number_of_times90_days_late'])
y = 'serious_dlqin2yrs'
models_to_run=['RF','LR','DT','KNN','NB','AB','BAG']
grid_type = 'standard'

def main(filename):
	df = explore(filename)
	df = pre.impute(df, ['monthly_income'], method = 'mean')
	pre.save_clean(df)

	clfs, grid = cl.define_clfs_params(grid_type)
	cl.clf_loop(models_to_run, clfs, grid, df[X], df[y])

if __name__=="__main__":
	if len(sys.argv) != 2:
		print('Input format: python pa3.py <filename path>') 
		sys.exit(1)

	filename = sys.argv[1]
	main(filename)
