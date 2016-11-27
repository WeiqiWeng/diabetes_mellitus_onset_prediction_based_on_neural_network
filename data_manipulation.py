import pandas as pd
import numpy as np
from sklearn import preprocessing
from imblearn.combine import SMOTETomek


df = pd.read_csv('../data/diabetes.csv')

label = np.array(df['Outcome'])

df = preprocessing.scale(df)
df[:, -1] = label

X = df[:, 0:-1]
Y = df[:, -1]

print('current proportion of y = 1: ', np.sum(Y)/len(Y))

sm = SMOTETomek(ratio=0.95, k=7)
X_resampled, Y_resampled = sm.fit_sample(X, Y)

print('after balancing, sample size = %d, feature size = %d' % X_resampled.shape)
print('after balancing, proportion of y = 1: ',np.sum(Y_resampled)/len(Y_resampled))

df = np.hstack((X_resampled, Y_resampled.reshape((len(Y_resampled), 1))))

pd.DataFrame(df).to_pickle('../data/diabetes_normalized_balanced.pickle')

