import sys
import numpy
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from keras import backend as K
import tensorflow as T
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam


seed = 7
numpy.random.seed(seed)

K.is_nan = T.is_nan
K.where = T.where

def mse(y_true, y_pred):
   cost = K.abs(y_pred - y_true)
   costs = K.where(K.is_nan(cost), T.zeros_like(cost), cost)
   return K.sum(costs, axis=-1)


def build_model(X_train, y_train, input_dim):
	model = Sequential()
	model = Sequential([
        Dense(500, input_dim=input_dim, kernel_initializer='normal', activation='relu'),
        Dense(300, kernel_initializer='normal', activation='relu'),
        Dense(100, kernel_initializer='normal', activation='relu'),
        Dense(1, kernel_initializer='normal', activation='linear')])
	model.compile(loss=mse, optimizer=Adam(0.0001))
	model.fit(X_train, y_train, epochs=20, batch_size=32)
	return model


training = '../data/scaffold_split/train_fold_4.csv'
test = '../data/scaffold_split/test_fold_4.csv'

tasks = 59 # total tasks

task_index = tasks + 1
print('tasks: %s' % tasks)

# load training train_set
train_set = pd.read_csv(training, delimiter=',', low_memory=False)
print('training data loaded')
#print(train_set.shape)

task_list = list(train_set.iloc[:,1:task_index].columns)
id_col = train_set.columns[0]

tasks_completed = []

# iterate over individual tasks and create single task models
print('\nbuilding models')
for task in task_list:
	tasks_remaining = len(task_list) - len(tasks_completed)
	print("current task: %s (%s remaining)" % (task, tasks_remaining))
	cols = list(train_set.iloc[:,task_index:].columns)
	cols.insert(0,id_col)
	cols.insert(1,task)
	dft = train_set[cols]
	# remove all rows with missing LD50 value for the current task
	dft = dft[pd.notnull(dft[task])]
	X_train = dft.iloc[:,2:].values
	y_train = dft.iloc[:,1:2].values
	input_dim = X_train.shape[1]

	model = build_model(X_train, y_train, input_dim)
	filename = '../models/st_dl/'+task+'.sav'
	pickle.dump(model, open(filename, 'wb'))
	tasks_completed.append(task)

print('finished building models\n')

# load training train_set
test_set = pd.read_csv(test, delimiter=',', low_memory=False)
print('test data loaded')
#print(test_set.shape)

print('\nstarted predictions\n')
print('Task\tr2\tmae\tmse\trmse')

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

	#print('loading model: ' + task)
	modelfile = task.replace('/', '_')
	filename = '../models/st_dl/'+modelfile+'.sav'
	model = pickle.load(open(filename, 'rb'))

	preds = model.predict(X_test)
	
	y_true = y_test
	y_pred = preds

	r2 = r2_score(y_true, y_pred)
	mae = mean_absolute_error(y_true, y_pred)
	mse = mean_squared_error(y_true, y_pred)
	rmse = sqrt(mean_squared_error(y_true, y_pred))
	print(task + "\t" + "%.2f\t%.2f\t%.2f\t%.2f" % (r2, mae, mse, rmse))

print('\nfinished predictions')
