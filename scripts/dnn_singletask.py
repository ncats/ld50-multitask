import sys
import timeit
import numpy
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam
import csv
import tensorflow as T
from math import sqrt
from statistics import mean, stdev
T.compat.v1.logging.set_verbosity(T.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

seed = 7
numpy.random.seed(seed)

K.is_nan = T.is_nan
K.where = T.where

def mse(y_true, y_pred):
   cost = K.abs(y_pred - y_true)
   costs = K.where(K.is_nan(cost), T.zeros_like(cost), cost)
   return K.sum(costs, axis=-1)


def build_model(X_train, y_train, input_dim, train_size, lrate):

	nodes = []

	if(train_size>700):
		nodes = [2000, 1000, 100]
	else:
		nodes = [500, 300, 100]

	model = Sequential()
	model = Sequential([
        Dense(nodes[0], input_dim=input_dim, kernel_initializer='normal', activation='relu'),
        #BatchNormalization(),
        Dense(nodes[1], kernel_initializer='normal', activation='relu'),
        #BatchNormalization(),
        Dense(nodes[2], kernel_initializer='normal', activation='relu'),
        #BatchNormalization(),
        Dense(1, kernel_initializer='normal', activation='linear')])
	model.compile(loss=mse, optimizer=Adam(lrate))
	model.fit(X_train, y_train, epochs=20, batch_size=32)
	return model

# dictionary file containing learning rates for the individual tasks
learn_rate_dict = 'learning_rates.csv'

df = pd.read_csv(learn_rate_dict, delimiter=',')

tnames = df['task'].tolist()
lrates = df['learning_rate'].tolist()

lr_dict = {tnames[i]: lrates[i] for i in range(len(tnames))}


training = '../data/scaffold_split/train_fold_4.csv'
test = '../data/scaffold_split/test_fold_4.csv'
tasks = 59

task_index = tasks + 1
print('tasks: %s' % tasks)

# load train_set
train_set = pd.read_csv(training, delimiter=',', low_memory=False)
train_set.drop('SMILES', axis=1, inplace=True)
print('training data loaded')
print(train_set.shape)

task_list = list(train_set.iloc[:,1:task_index].columns)
id_col = train_set.columns[0]

tasks_completed = []

# iterate over training set tasks and create single task models
print('\nbuilding models')
for task in task_list:

	lrate = 0.0001
	if task in lr_dict.keys():
		lrate = lr_dict[task]

	
	tasks_remaining = len(task_list) - len(tasks_completed)
	print("current task: %s (%s remaining)" % (task, tasks_remaining))
	cols = list(train_set.iloc[:,task_index:].columns)
	cols.insert(0,id_col)
	cols.insert(1,task)
	dft = train_set[cols]
	# remove all rows with missing LD50 value for the current task
	dft = dft[pd.notnull(dft[task])]
	X_train = dft.iloc[:,2:].values
	#print(X_train.shape)
	y_train = dft.iloc[:,1:2].values
	#print(y_train.shape)
	input_dim = X_train.shape[1]

	model = build_model(X_train, y_train, input_dim, X_train.shape[0], lrate)
	filename = '../models/st_dl/fold_4/'+task+'.sav'
	pickle.dump(model, open(filename, 'wb'))
	tasks_completed.append(task)
	
print('finished building models\n')

# load test_set
test_set = pd.read_csv(test, delimiter=',', low_memory=False)
test_set.drop('SMILES', axis=1, inplace=True)
print('test data loaded')
print(test_set.shape)

print('\nstarted predictions\n')

pred_df = test_set.iloc[0:0]
pred_df = pred_df.iloc[:, 0:2]
preds = []
pred_df.insert(loc=2, column='pred_LD50', value=preds)

# iterate over test set, load single task models and predict test set 
for task in task_list:
	
	cols = list(test_set.iloc[:,task_index:].columns)
	cols.insert(0,id_col)
	cols.insert(1,task)
	dft = test_set[cols]
	# remove all rows with missing LD50 value for the current task
	dft = dft[pd.notnull(dft[task])]
	X_test = dft.iloc[:,2:].values
	y_test = dft.iloc[:,1:2].values

	print('loading model: ' + task)
	modelfile = task.replace('/', '_')
	filename = '../models/st_dl/fold_4/'+modelfile+'.sav'
	model = pickle.load(open(filename, 'rb'))
	preds = model.predict(X_test)

	dft.insert(loc=2, column='pred_LD50', value=preds)
	pdft = dft.iloc[:, 0:3]
	pdft2 = pdft.rename({task: 'LD50'}, axis='columns')
	pdft2['Task'] = task
	#print(pdft.head())
	pred_df = pred_df.append(pdft2, sort=False)

print('\nfinished predictions')

y_true = pred_df['LD50'].values
y_pred = pred_df['pred_LD50'].values

# calculate performance task-wise
r2_scores = []
mae_scores = []
mse_scores = []
rmse_scores = []

print("\ntask-wise performance")
print('\ntask\tr2\tmae\tmse\trmse')
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
