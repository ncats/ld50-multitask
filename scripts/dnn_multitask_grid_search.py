import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Activation, BatchNormalization, Average
from keras.optimizers import SGD
from keras.optimizers import Adam, Adadelta
from keras.wrappers.scikit_learn import KerasRegressor

import numpy
import tensorflow as T
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from statistics import mean, stdev

seed = 7
numpy.random.seed(seed)

# the code is adapted from
# 1. https://medium.com/@am.benatmane/keras-hyperparameter-tuning-using-sklearn-pipelines-grid-search-with-cross-validation-ccfc74b0ce9f
# 2. https://github.com/keras-team/keras/blob/master/examples/mnist_sklearn_wrapper.py

# custom loss function for missing values in input data (i.e. target labels or values)
K.is_nan = T.is_nan
K.where = T.where

def mse(y_true, y_pred):
   cost = K.abs(y_pred - y_true)
   costs = K.where(K.is_nan(cost), T.zeros_like(cost), cost)
   return K.sum(costs, axis=-1)

# read training data
dataset = pd.read_csv('../data/grid_search/train_scf_split.csv', delimiter=',', low_memory=False)

# split into feature (X) and target (y) variables
X = dataset.iloc[:,180:1204].values # avalon fp 1024 bits
y = dataset.iloc[:,2:61].values # values for 59 endpoints

print(X.shape)
print(y.shape)

# functions to create keras models
def create_model_v1(activation='relu', learn_rate=0.01):
    model = Sequential()
    model.add(Dense(1500, kernel_initializer='uniform', activation=activation))
    model.add(Dense(500, kernel_initializer='uniform', activation=activation))
    model.add(Dense(100, kernel_initializer='uniform', activation=activation))
    model.add(Dense(y.shape[1], kernel_initializer='uniform', activation='linear'))
    optimizer = Adam(learn_rate)
    model.compile(loss=mse, optimizer=optimizer, metrics=['accuracy'])
    return model

def create_model_v2(dense_layer_sizes, activation='relu', learn_rate=0.01):
    model = Sequential()
    model.add(Dense(2000, input_dim=X.shape[1], activation=activation, kernel_initializer='normal'))
    for layer_size in dense_layer_sizes:
        model.add(Dense(layer_size, activation=activation, kernel_initializer='normal'))
    model.add(Dense(y.shape[1], activation='linear', kernel_initializer='normal'))
    optimizer = Adam(learn_rate)
    model.compile(loss=mse, optimizer=optimizer, metrics=['accuracy'])
    return model

dense_size_candidates = [[700, 500], [500, 100]]

# keras estimator for grid search
kears_estimator = KerasRegressor(build_fn=create_model_v1, verbose=1)

#Grid search and parameters
estimator = Pipeline([("kr", kears_estimator)])

param_grid = {
    'kr__learn_rate':[0.01, 0.001, 0.0001, 0.00001],
    'kr__activation':['relu'], # activation functions (other functions: sigmoid, linear, softmax, swish etc.)
#   'kr__dense_layer_sizes':dense_size_candidates, # if dense layers are to be optimized, use create_model_v2 instead of create_model_v1
    'kr__epochs':[20, 50], # epochs is available for tuning even when not an argument to model building function
    'kr__batch_size':[128,256,512,1024]
}

kfolds = 2

grid = GridSearchCV(estimator=estimator,
                    n_jobs=1,
                    verbose=2,
                    return_train_score=True,
                    cv=kfolds,
                    param_grid=param_grid,)

grid_result = grid.fit(X, y, )

# best parameters
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']

# print and write full grid results to file
outfile = open('../grid_cv_results/full_grid_results.txt', 'w')
outfile.write("Best: %f using %s\n" % (grid_result.best_score_, grid_result.best_params_))

for mean, stdev, param in zip(means, stds, params):
    # print("%f (%f) with: %r" % (mean, stdev, param))
    outfile.write("%f (%f) with: %r\n" % (mean, stdev, param))

outfile.close()
