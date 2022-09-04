import pandas as pd
import numpy as np
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l1_l2
import matplotlib.pyplot as plt
import pandas as pd
import math
import random
import scipy.stats as stats
import pylab as pl
import pmdarima
from pmdarima.arima import auto_arima, ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
from tqdm import tqdm

class Double_Tanh(Activation):
    def __init__(self, activation, **kwargs):
        super(Double_Tanh, self).__init__(activation, **kwargs)
        self.__name__ = 'double_tanh'

def double_tanh(x):
    return (K.tanh(x) * 2)

"""df = pd.read_csv('backtest_df.csv')
df = df.set_index(['Date'])[:800]

portfolio = list(df.columns.values)

index_list = []
for _ in range(100):
    indices = []
    for k in range(_, 600, 100):
        indices.append(k)
    index_list.append(indices)

def rolling_corr(df, item1, item2):
    df_pair = pd.concat([df[item1], df[item2]], axis=1)
    df_pair.columns = [item1, item2]
    df_corr = df_pair[item1].rolling(window=100).corr(df_pair[item2])
    return df_corr

data_matrix = []
count = 0
for i in range(8):
    for j in range(7-i):
        a = portfolio[i]
        b = portfolio[7-j]
        file_name = a + '_' + b

        corr_series = rolling_corr(df, a, b)[99:]
        corr_series.index = [i for i in range(len(corr_series))]
        for _ in range(100):
            corr_strided = list(corr_series[index_list[_]][:24]).copy()
            data_matrix.append(corr_strided)
            count+=1
                
data_matrix = np.transpose(data_matrix)
data_dictionary = {}
for i in range(len(data_matrix)):
    data_dictionary[str(i)] = data_matrix[i]
data_df = pd.DataFrame(data_dictionary)

data_df.to_csv('dataset_backtest.csv')

df = pd.read_csv('dataset_backtest.csv')
df = df.loc[:, df.columns.str.contains('^Unnamed')]

num_list = []
for i in range(6):
    num_list.append(str(i))
data_df = data_df[num_list].copy()
data_df = np.transpose(data_df)

indices = [20*k for k in range(140)]
data_df = pd.DataFrame(data_df[indices])

train = []

for i in range(data_df.shape[1]):
    tmp = data_df[20*i].copy()
    train.append(tmp[:21])
    
train = pd.DataFrame(train)
train.to_csv('train_backtest.csv')
train = pd.read_csv('train_test_set/train.csv')
train = np.transpose(train.loc[:,~train.columns.str.contains('^Unnamed')])

datasets = [train]
model_110 = ARIMA(order=(1,1,0), suppress_warnings=True)
model_011 = ARIMA(order=(0,1,1), suppress_warnings=True)
model_111 = ARIMA(order=(1,1,1), suppress_warnings=True)
model_211 = ARIMA(order=(2,1,1), suppress_warnings=True)
model_210 = ARIMA(order=(2,1,0), suppress_warnings=True)

prediction = []
train_X = []; train_Y = []


flag = 0

for i in tqdm(range(140)):
    tmp = []
    c=0
    for s in datasets :
        c+=1
        try:
            model1 = model_110.fit(s[i])
            model = model1
            
            try:
                model2 = model_011.fit(s[i])
                
                if model.aic() <= model2.aic() :
                    pass
                else :
                    model = model2
                    
                try :
                    model3 = model_111.fit(s[i])
                    if model.aic() <= model3.aic() :
                        pass
                    else :
                        model = model3
                except :
                    try:
                        model4 = model_211.fit(s[i])
                        
                        if model.aic() <= model4.aic() :
                            pass
                        else:
                            model = model4
                    except:
                        try:
                            model5 = model_210.fit(s[i])
                            
                            if model.aic() <= model5.aic():
                                pass
                            else :
                                model = model5
                        except :
                            pass
                    
            except:
                try:
                    model3 = model_111.fit(s[i])

                    if model.aic() <= model3.aic() :
                        pass
                    else :
                        model = model3
                except :
                    try:
                        model4 = model_211.fit(s[i])
                        
                        if model.aic() <= model4.aic() :
                            pass
                        else:
                            model = model4
                    except:
                        try:
                            model5 = model_210.fit(s[i])
                            
                            if model.aic() <= model5.aic():
                                pass
                            else :
                                model = model5
                        except :
                            pass
                
        except:
            try:
                model2 = model_011.fit(s[i])
                model = model2
            
                try :
                    model3 = model_111.fit(s[i])
                    
                    if model.aic() <= model3.aic():
                        pass
                    else:
                        model = model3
                except :
                    try:
                        model4 = model_211.fit(s[i])
                        
                        if model.aic() <= model4.aic() :
                            pass
                        else:
                            model = model4
                    except:
                        try:
                            model5 = model_210.fit(s[i])
                            
                            if model.aic() <= model5.aic():
                                pass
                            else :
                                model = model5
                        except :
                            pass
            
            except :
                try:
                    model3 = model_111.fit(s[i])
                    model = model3
                except :
                    try:
                        model4 = model_211.fit(s[i])
                        
                        if model.aic() <= model4.aic() :
                            pass
                        else:
                            model = model4
                    except:
                        try:
                            model5 = model_210.fit(s[i])
                            
                            if model.aic() <= model5.aic():
                                pass
                            else :
                                model = model5
                        except :
                            flag = 1
                            print(str(c) + " FATAL ERROR")
                            break
        
        predictions = list(model.predict_in_sample())
        prediction.append(predictions)
        #pad the first time step of predictions with the average of the prediction values
        #so as to match the length of the s[i] data
        
        residual = pd.Series(np.array(s[i]) - np.array(predictions))
        tmp.append(np.array(residual))
        
                    
    if flag == 1:
        break
    train_X.append(tmp[0][:20])
    train_Y.append(tmp[0][20])

pd.DataFrame(train_X).to_csv('train_X_backtest.csv')
train = pd.read_csv('train_X_backtest.csv')
train = np.transpose(train.loc[:,~train.columns.str.contains('^Unnamed')])
train_melt = sorted(np.array(train.melt()['value']))
fit = stats.norm.pdf(train_melt, np.mean(train_melt), np.std(train_melt))
"""
# -------------------------------------------------------------------
get_custom_objects().update({'double_tanh':Double_Tanh(double_tanh)})
train_X= pd.read_csv('train_X_backtest.csv')
train_X = train_X.loc[:, ~train_X.columns.str.contains('^Unnamed')]

recent_model_name = 'r_epoch124.h5'
d = 'hybrid_LSTM'
filepath = 'models/' + d + '/' + recent_model_name
model = load_model(filepath)

train_X = np.asarray(train_X).reshape((140, 20, 1))

lstm = model.predict(train_X).ravel()

print(np.shape(lstm))

# -------------------------------------------------------------------

"""corr = lstm.copy()

corr_matrix = np.identity(n=8)
cnt = 0
for i in range(8):
    for j in range(i, 8):
      if i == j:
        pass
      else:
        corr_matrix[i,j] = corr[cnt]
        corr_matrix[j,i] = corr[cnt]
        cnt += 1

np.save('corr_matrix.npy', corr_matrix)
"""
