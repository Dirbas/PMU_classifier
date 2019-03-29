# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 11:53:24 2019

@author: ago
"""

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from scipy import stats
from IPython.display import display, HTML

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import LSTM



def gen_features(X):
    strain = []
    strain.append(X.mean())
    strain.append(X.std())
    strain.append(X.min())
    strain.append(X.max())
    strain.append(X.kurtosis())
    strain.append(X.skew())
    return pd.Series(strain)

df = pd.read_csv('frequency.csv', delimiter=';')
df.rename(columns={'Freq phase A': 'value a', 'Freq phase B': 'value b'}, inplace=True)
df['label'] = 'Frequency'
df = df.dropna()
print(df)


df1 = pd.read_csv('V RMS.csv', delimiter=';')
df1.rename(columns={'V RMS phase A': 'value a', 'V RMS phase B': 'value b'}, inplace=True)
del df1['V RMS phase C']
df1['label'] = 'V RMS'
df1 = df1.dropna()
print(df1)

df2 = pd.read_csv('I_phase.csv', delimiter=';')
df2.rename(columns={'I ph.angle phase A': 'value a', 'I ph.angle phase B': 'value b'}, inplace=True)
del df2['I ph.angle phase C']
df2['label'] = 'I ph.angle'
df2 = df2.dropna()
print(df2)


df3 = pd.read_csv('Vph angle.csv', delimiter=';')
df3.rename(columns={'V ph.angle phase A': 'value a', 'V ph.angle phase B': 'value b'}, inplace=True)
del df3['V ph.angle phase C']
df3['label'] = 'V ph.angle'
df3 = df3.dropna()
print(df3)

df4 = pd.read_csv('I_rms.csv', delimiter=';')
df4.rename(columns={'I RMS phase A': 'value a', 'I RMS phase B': 'value b'}, inplace=True)
del df4['I RMS phase C']
df4['label'] = 'I RMS'
df4 = df4.dropna()
print(df4)

verticalStack = pd.concat([df,df1,df2,df4], axis=0)
print(verticalStack)
# Write DataFrame to CSV
verticalStack.to_csv('PMU_data.csv')

def plot_activity(label, verticalStack):
    data = verticalStack[verticalStack['label'] == label][['value a', 'value b']][:200]
    axis = data.plot(subplots=True, figsize=(16, 12), 
                     title=label)
    for ax in axis:
        ax.legend(loc='lower left', bbox_to_anchor=(1.0, 0.5))

plot_activity("Frequency", df)
plot_activity("V RMS", df1)
plot_activity("I ph.angle", df2)
plot_activity("V ph.angle", df3)
plot_activity("I RMS", df4)