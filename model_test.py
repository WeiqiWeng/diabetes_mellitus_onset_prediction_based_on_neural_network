import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sknn import ae, mlp
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = np.array(pd.read_pickle('../data/diabetes_test.pickle'))
X = df[:, 0:-1]
Y = df[:, -1]

pca_config = pickle.load(open( "../data/pca_config.pickle", "rb" ))
pca_config = pca_config[5]

n_component = pca_config['n_components']
hidden_layer_size = pca_config['param']['hidden_layer_sizes']
weight_decay = pca_config['param']['alpha']

pca = PCA(n_components=n_component)
pca_X = pca.fit_transform(X)

nn = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=weight_decay, hidden_layer_sizes=(hidden_layer_size))
nn.fit(pca_X, Y)
print('accuracy on test set: ', nn.score(pca_X, Y))

df = np.array(pd.read_pickle('../data/diabetes_normalized_balanced.pickle'))
X = df[:, 0:-1]
Y = df[:, -1]
pca = PCA(n_components=n_component)
pca_X = pca.fit_transform(X)
nn.fit(pca_X, Y)
print('accuracy on the whole data set: ', nn.score(pca_X, Y))