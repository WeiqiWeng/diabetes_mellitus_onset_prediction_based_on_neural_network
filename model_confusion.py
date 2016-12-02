import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sknn import ae, mlp

# Train model (PCA)
df = np.array(pd.read_pickle('../data/diabetes_train.pickle'))
X = df[:, 0:-1]
Y = df[:, -1]

pca_config = pickle.load(open( "../data/pca_config.pickle", "rb" ))
pca_config = pca_config[5]

n_component = pca_config['n_components']
print(n_component)
hidden_layer_size = pca_config['param']['hidden_layer_sizes']
print(hidden_layer_size)
weight_decay = pca_config['param']['alpha']
print(weight_decay)

pca = PCA(n_components=n_component)
pca_X = pca.fit_transform(X)

nn = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=weight_decay, hidden_layer_sizes=(hidden_layer_size))
nn.fit(pca_X, Y)

# Predict
testFile = np.array(pd.read_pickle('../data/diabetes_test.pickle'))
X = testFile[:, 0:-1]
Y = testFile[:, -1]

tst_X = pca.transform(X)
result = nn.predict(tst_X)


confusion = [[0, 0], [0, 0]]
for i in range(len(Y)):
    if (Y[i] > 0.0 and result[i] > 0.0):
        confusion[1][1] += 1
    elif (Y[i] > 0.0 and result[i] < 1.0):
        confusion[1][0] += 1
    elif (Y[i] < 1.0 and result[i] > 0.0):
        confusion[0][1] += 1
    elif (Y[i] < 1.0 and result[i] < 1.0):
        confusion[0][0] += 1

print(confusion)
print(Y)
print(result)