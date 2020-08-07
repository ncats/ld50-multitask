# This is a modified version of the DLCA architecture previously proposed by Zakharov et al.

# More details about DLCA architecture can be found at https://doi.org/10.1021/acs.jcim.9b00526

import keras
from keras import backend as K
from keras import losses
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Activation, BatchNormalization, Average, Embedding, Bidirectional, concatenate, Flatten
from keras.optimizers import SGD
from keras.optimizers import Adam, Adadelta, Adamax, Nadam, Adagrad
from keras.layers import Conv1D, GlobalMaxPooling1D, MaxPooling1D
from keras.layers import LSTM
import numpy
import csv
import tensorflow as T
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt
from statistics import mean, stdev
from sklearn.utils import class_weight
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention
from sklearn.preprocessing import StandardScaler

seed = 7
numpy.random.seed(seed)

dataset = pd.read_csv('../data/scaffold_split/train_fold_4.csv', delimiter=',', low_memory=False)
# split into input (X) and output (Y) variables
y_train = dataset.iloc[:,2:61].values

scaler = StandardScaler()

X_train1 = dataset.iloc[:,1204:2228].values # morgan fp
X_train2 = dataset.iloc[:,180:1204].values # avalon fp
X_train3 = dataset.iloc[:,2228:3252].values # atom pair fp
X_train4 = dataset.iloc[:,61:180].values # rdkit desc
X_train4 = scaler.fit_transform(X_train4)
X_train5 = dataset.iloc[:,1:2].values # smiles


for p in range (X_train5.shape[0]):
  s = X_train5[p,0]
  s = s.replace("[nH]","A")
  s = s.replace("Cl","L")
  s = s.replace("Br","R")
  s = s.replace("[C@]","C")
  s = s.replace("[C@@]","C")
  s = s.replace("[C@@H]","C")
  s =[s[i:i+1] for i in range(0,len(s),1)]
  s = " ".join(s)
  X_train5[p,0] = s
X_train5 = X_train5[:,0]
X_train5 = X_train5.tolist()


tokenizer = Tokenizer(num_words=100, filters='!"$%&*+,-./:;<>?\\^_`{|}~\t\n')
tokenizer.fit_on_texts(X_train5)
X_train5 = tokenizer.texts_to_sequences(X_train5)
X_train5 = pad_sequences(X_train5, maxlen=200, padding='post')



dataset = pd.read_csv('../data/scaffold_split/test_fold_4.csv', delimiter=',', low_memory=False)

task_list = list(dataset.iloc[:,2:61].columns)

# split into input (X) and output (Y) variables
y_test = dataset.iloc[:,2:61].values

# different descriptors
X_test1 = dataset.iloc[:,1204:2228].values # morgan fp
X_test2 = dataset.iloc[:,180:1204].values # avalon fp
X_test3 = dataset.iloc[:,2228:3252].values # atom pair fp
X_test4 = dataset.iloc[:,61:180].values # rdkit desc
X_test4 = scaler.transform(X_test4)
X_test5 = dataset.iloc[:,1:2].values # smiles

for p in range (X_test5.shape[0]):
  s = X_test5[p,0]
  s = s.replace("[nH]","A")
  s = s.replace("Cl","L")
  s = s.replace("Br","R")
  s = s.replace("[C@]","C")
  s = s.replace("[C@@]","C")
  s = s.replace("[C@@H]","C")
  s =[s[i:i+1] for i in range(0,len(s),1)]
  s = " ".join(s)
  X_test5[p,0] = s
X_test5 = X_test5[:,0]
X_test5 = X_test5.tolist()
X_test5 = tokenizer.texts_to_sequences(X_test5)
X_test5 = pad_sequences(X_test5, maxlen=200, padding='post')


desc1 = Input(shape=(X_train1.shape[1],),name='desc1')
desc2 = Input(shape=(X_train2.shape[1],),name='desc2')
desc3 = Input(shape=(X_train3.shape[1],),name='desc3')
desc4 = Input(shape=(X_train4.shape[1],),name='desc4')
desc5 = Input(shape=(200,),name='desc5')

desc1 = Input(shape=(X_train1.shape[1],))
x = Dense(8000, activation='relu')(desc1)
x = Dense(2000, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
x = Dense(700, activation='relu')(x)
out1 = Dense(59, activation='linear', name='out1')(x)

desc2 = Input(shape=(X_train2.shape[1],))
x2 = Dense(8000, activation='relu')(desc2)
x2 = Dense(2000, activation='relu')(x2)
x2 = Dense(1000, activation='relu')(x2)
x2 = Dense(700, activation='relu')(x2)
out2 = Dense(59, activation='linear', name='out2')(x2)

desc3 = Input(shape=(X_train3.shape[1],))
x3 = Dense(8000, activation='relu')(desc3)
x3 = Dense(2000, activation='relu')(x3)
x3 = Dense(1000, activation='relu')(x3)
x3 = Dense(700, activation='relu')(x3)
out3 = Dense(59, activation='linear', name='out3')(x3)

desc4 = Input(shape=(X_train4.shape[1],))
x4 = Dense(8000, activation='relu')(desc4)
Dropout(0.3)
BatchNormalization(),
x4 = Dense(2000, activation='relu')(x4)
x4 = Dense(1000, activation='relu')(x4)
BatchNormalization(),
x4 = Dense(700, activation='relu')(x4)
out4 = Dense(59, activation='linear', name='out4')(x4)

desc5 = Input(shape=(200,))
x5 = Embedding(200, 128, input_length=200)(desc5)
x5 = Conv1D(256,16,padding='valid',activation='relu',strides=1)(x5)
x5 = GlobalMaxPooling1D()(x5)
x5 = Dense(200, activation='relu')(x5)
out5 = Dense(59, activation='linear',name='out5')(x5)

out6 = keras.layers.average([out1, out2, out3, out4, out5])

model = Model(inputs=[desc1, desc2, desc3, desc4, desc5], outputs=[out1, out2, out3, out4, out5, out6])


K.is_nan = T.math.is_nan                       
K.where = T.where

def mse(y_true, y_pred):
     y_true = K.where(K.is_nan(y_true), y_pred, y_true)
     cost = K.abs(y_pred - y_true)
     return K.sum(cost, axis=-1)


sgd = SGD(lr=0.03, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss=mse,
              optimizer=Adagrad(0.01))

print (model.summary())              #optimizer=sgd)


model.fit([X_train1, X_train2, X_train3, X_train4, X_train5], [y_train,y_train,y_train,y_train,y_train,y_train],
          epochs=20,
          batch_size=128)

score = model.evaluate([X_test1, X_test2, X_test3, X_test4, X_test5], [y_test,y_test,y_test,y_test,y_test,y_test], batch_size=128)


predictions = model.predict([X_test1,X_test2, X_test3, X_test4, X_test5])

predictions = numpy.asarray(predictions)

dataset.drop('SMILES', axis=1, inplace=True)

df_all_preds = dataset.iloc[:,0:60]
df_all_preds = pd.melt(df_all_preds, id_vars='RTECS_ID', value_vars=task_list, var_name='Task', value_name='LD50')

for i in range(len(predictions)):
    preds = predictions[i]

    df_test = dataset.iloc[:,0:60] # used for peformance assessment

    # transform predictions to df
    df_pred =pd.DataFrame(data=preds[0:,0:], index=[i for i in range(preds.shape[0])], columns=task_list)
    rtecs_ids = df_test['RTECS_ID'].values
    df_pred = df_pred.assign(RTECS_ID = rtecs_ids)

    # reshape df_test and df_pred
    df_test = pd.melt(df_test, id_vars='RTECS_ID', value_vars=task_list, var_name='Task', value_name='LD50')
    df_pred = pd.melt(df_pred, id_vars='RTECS_ID', value_vars=task_list, var_name='Task', value_name='pred_LD50')

    # get pred column and append to all preds df
    y_pred_wnan = df_pred['pred_LD50'].values

    pred_col = 'pred_LD50_' + str(i)
    df_all_preds[pred_col] = y_pred_wnan

    # merge df_test and df_pred
    final_df = pd.merge(df_test, df_pred,  how='left', left_on=['RTECS_ID','Task'], right_on = ['RTECS_ID','Task'])
    final_df = final_df[pd.notnull(final_df['LD50'])]


    y_true = final_df['LD50'].values
    y_pred = final_df['pred_LD50'].values

    # calculate performance task-wise
    r2_scores = []
    mae_scores = []
    mse_scores = []
    rmse_scores = []

    print("\ntask-wise performance")
    for task, dft in final_df.groupby('Task'):
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
      # print(task + "\t" + "%.2f\t%.2f\t%.2f\t%.2f" % (r2, mae, mse, rmse))

    print("prediction for:\t%s"% i)
    print("\ntask-wise average performance")
    print("metric\tavg\tstdev")
    print("r^2\t%.3f\t%.3f" % (mean(r2_scores), stdev(r2_scores)))
    print("mae\t%.3f\t%.3f" % (mean(mae_scores), stdev(mae_scores)))
    print("mse\t%.3f\t%.3f" % (mean(mse_scores), stdev(mse_scores)))
    print("rmse\t%.3f\t%.3f" % (mean(rmse_scores), stdev(rmse_scores)))

df_all_preds.to_csv('predictions_scaffold_split_fold_4_all.csv' , index = None, header=True) # if you want all 59 predictions for each molecule

df_all_preds = df_all_preds[pd.notnull(df_all_preds['LD50'])]
df_all_preds.to_csv('predictions_scaffold_split_fold_4.csv' , index = None, header=True) # if you need only those predictions for which you have original LD50 value

