import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


df = pd.read_pickle('../data/diabetes_normalized_balanced.pickle')

train, test = train_test_split(df, test_size = 0.21)
print('training data set size: ',len(train))
validation, test = train_test_split(test, test_size = 0.5)
print('validation data set size: ',len(validation))
print('test data set size: ',len(test))

train.to_pickle('../data/diabetes_train.pickle')
validation.to_pickle('../data/diabetes_validation.pickle')
test.to_pickle('../data/diabetes_test.pickle')