import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from statistics import mean, stdev


def melt_dataset(path_to_file, tasks):
	'''
	reads in the dataset and reshapes it for building baselines models
	'''
	print('reading dataset...')
	dataset = pd.read_csv(path_to_file, delimiter=',', low_memory=False)
	dataset.drop('SMILES', axis=1, inplace=True)
	task_index = tasks + 1
	task_list = list(dataset.iloc[:,1:task_index].columns)
	id_df = dataset.drop(task_list, axis=1)
	id_list = list(id_df.columns)

	dataset = pd.melt(dataset, id_vars=id_list, value_vars=task_list, var_name='Task', value_name='LD50')
	dataset = dataset[pd.notnull(dataset['LD50'])]

	cols = list(dataset.columns)
	cols = [cols[0]] + [cols[-2]] + [cols[-1]] + cols[1:-2]
	dataset = dataset[cols]

	return dataset


training = '../data/scaffold_split/train_fold_4.csv'
test = '../data/scaffold_split/test_fold_4.csv'

tasks = 59 # total tasks

train_df = melt_dataset(training, tasks)
print('training data loaded')
#print(train_df.shape)

test_df = melt_dataset(test, tasks)
print('test data loaded')
#print(test_df.shape)


print('\nbuilding models')
for target, dft in train_df.groupby('Task'):
	print("current model: " + target)
	X_train = dft.iloc[:,3:]
	y_train = dft['LD50'].values
	target = target.replace('/', '_')
	model = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs=-1)
	model.fit(X_train, y_train)
	filename = '../models/st_rf/scaffold/fold_4/'+target+'.sav'
	pickle.dump(model, open(filename, 'wb'))

print('finished building models\n')


print('started predictions')
pred_df = test_df.iloc[0:0]
pred_df = pred_df.iloc[:, 0:3]
preds = []
pred_df.insert(loc=3, column='pred_LD50', value=preds)

for target, dft in test_df.groupby('Task'):
	X_test = dft.iloc[:,3:]
	#y_test = dft['pLD50'].values
	print('loading model: ' + target)
	target = target.replace('/', '_')
	filename = '../models/st_rf/scaffold/fold_4/'+target+'.sav'
	model = pickle.load(open(filename, 'rb'))
	preds = model.predict(X_test)
	dft.insert(loc=3, column='pred_LD50', value=preds)
	pdft = dft.iloc[:, 0:4]
	#print(pdft.head())
	pred_df = pred_df.append(pdft, sort=False)

print('finished predictions\n')

# file to save predictions
predfile = '../predictions/scaffold/predictions_RF_fold_4.csv'

pred_df.to_csv(predfile, sep=',', encoding='utf-8', index=False)
print('predictions written to: '+ predfile)


y_true = pred_df['LD50'].values
y_pred = pred_df['pred_LD50'].values

print("\noverall performance")
print("metric\tvalue")
print("r^2\t%.2f" % r2_score(y_true, y_pred))
print("mae\t%.2f" % mean_absolute_error(y_true, y_pred))
print("mse\t%.2f" % mean_squared_error(y_true, y_pred))
print("rmse\t%.2f" % sqrt(mean_squared_error(y_true, y_pred)))

# calculate performance task-wise
r2_scores = []
mae_scores = []
mse_scores = []
rmse_scores = []

print("\ntask-wise performance")
for task, dft in pred_df.groupby('Task'):
    y_true = dft['LD50'].values
    y_pred = dft['pred_LD50'].values

    r2 = r2_score(y_true, y_pred)
    r2_scores.append(r2)
    mae = mean_absolute_error(y_true, y_pred)
    mae_scores.append(mae)
    mse = mean_squared_error(y_true, y_pred)
    mse_scores.append(mse)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    rmse_scores.append(rmse)
    print(task + "\t" + "%.2f\t%.2f\t%.2f\t%.2f" % (r2, mae, mse, rmse))

# final performance
print("\ntask-wise average performance")
print("metric\tavg\tstdev")
print("r^2\t%.2f\t%.2f" % (mean(r2_scores), stdev(r2_scores)))
print("mae\t%.2f\t%.2f" % (mean(mae_scores), stdev(mae_scores)))
print("mse\t%.2f\t%.2f" % (mean(mse_scores), stdev(mse_scores)))
print("rmse\t%.2f\t%.2f" % (mean(rmse_scores), stdev(rmse_scores)))
