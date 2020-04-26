from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, BatchNormalization
from keras.layers.noise import AlphaDropout
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam
import numpy
import pandas as pd
import tensorflow as T
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from statistics import mean, stdev

seed = 7
numpy.random.seed(seed)

K.is_nan = T.is_nan
K.where = T.where

# custom loss function to handle missing values in the inpuit data
def mse(y_true, y_pred):
   cost = K.abs(y_pred - y_true)
   costs = K.where(K.is_nan(cost), T.zeros_like(cost), cost)
   return K.sum(costs, axis=-1)

training = '../data/scaffold_split/train_fold_4.csv'
test = '../data/scaffold_split/test_fold_4.csv'

tasks = 59
task_index = tasks + 1
print('tasks: %s' % tasks)

# load training dataset
dataset = pd.read_csv(training, delimiter=',', low_memory=False)
dataset.drop('SMILES', axis=1, inplace=True)
X_train = dataset.iloc[:,task_index:].values
y_train = dataset.iloc[:,1:task_index].values
print("loaded training data: %s, %s" % (X_train.shape, y_train.shape))

# load test dataset
dataset = pd.read_csv(test, delimiter=',', low_memory=False)
dataset.drop('SMILES', axis=1, inplace=True)
df_test = dataset.iloc[:,0:task_index] # used for peformance assessment
X_test = dataset.iloc[:,task_index:].values
y_test = dataset.iloc[:,1:task_index].values
print("loaded test data: %s, %s" % (X_test.shape, y_test.shape))

task_list = list(dataset.iloc[:,1:task_index].columns)

input_dim = X_train.shape[1]

model = Sequential()

model = Sequential([
        Dense(2000, input_dim=input_dim, kernel_initializer='normal', activation='selu'),
	#BatchNormalization(),
	Dropout(0.2),
        Dense(700, kernel_initializer='normal', activation='selu'),
        #BatchNormalization(),
	Dropout(0.2),
        Dense(500, kernel_initializer='normal', activation='selu'),
        #BatchNormalization(),
	Dropout(0.2),
        Dense(tasks, kernel_initializer='normal', activation='linear')])


model.compile(loss=mse, optimizer=Adam(0.0001))
model.fit(X_train, y_train, epochs=20, class_weight=y_train.sum(axis=0), batch_size=32)
score = model.evaluate(X_test, y_test, batch_size=32)
print("results: %.2f (%.2f) mse" % (score.mean(), score.std()))

predictions = model.predict(X_test)

# transform predictions to df
df_pred =pd.DataFrame(data=predictions[0:,0:], index=[i for i in range(predictions.shape[0])], columns=task_list)
rtecs_ids = df_test['RTECS_ID'].values
df_pred = df_pred.assign(RTECS_ID = rtecs_ids)

# reshape df_test and df_pred
df_test = pd.melt(df_test, id_vars='RTECS_ID', value_vars=task_list, var_name='Task', value_name='LD50')
df_pred = pd.melt(df_pred, id_vars='RTECS_ID', value_vars=task_list, var_name='Task', value_name='pred_LD50')

# merge df_test and df_pred
final_df = pd.merge(df_test, df_pred,  how='left', left_on=['RTECS_ID','Task'], right_on = ['RTECS_ID','Task'])
final_df = final_df[pd.notnull(final_df['LD50'])]
final_df.to_csv("predictions.csv", index = None, header=True)

y_true = final_df['LD50'].values
y_pred = final_df['pred_LD50'].values

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

for task, dft in final_df.groupby('Task'):
    y_true = dft['LD50'].values
    y_pred = dft['pred_LD50'].values
    
    r2_scores.append(r2_score(y_true, y_pred))
    mae_scores.append(mean_absolute_error(y_true, y_pred))
    mse_scores.append(mean_squared_error(y_true, y_pred))
    rmse_scores.append(sqrt(mean_squared_error(y_true, y_pred)))

# final performance
print("\ntask-wise performance")
print("metric\tavg\tstdev")
print("r^2\t%.2f\t%.2f" % (mean(r2_scores), stdev(r2_scores)))
print("mae\t%.2f\t%.2f" % (mean(mae_scores), stdev(mae_scores)))
print("mse\t%.2f\t%.2f" % (mean(mse_scores), stdev(mse_scores)))
print("rmse\t%.2f\t%.2f" % (mean(rmse_scores), stdev(rmse_scores)))