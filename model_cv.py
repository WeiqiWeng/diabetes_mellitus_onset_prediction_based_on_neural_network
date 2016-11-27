import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sknn import ae, mlp
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = np.array(pd.read_pickle('../data/diabetes_validation.pickle'))
X = df[:, 0:-1]
Y = df[:, -1]

pca_config = pickle.load(open( "../data/pca_config.pickle", "rb" ))
pca_config = pca_config[5]

n_component = pca_config['n_components']
hidden_layer_size = pca_config['param']['hidden_layer_sizes']
weight_decay = pca_config['param']['alpha']

pca = PCA(n_components=n_component)
pca_X = pca.fit_transform(X)

folds = range(5, 51, 5)

nn = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=weight_decay, hidden_layer_sizes=(hidden_layer_size))
scores_pca = []
std_pca = []
for n in folds:
    print('%d folds for PCA' % n)
    result = cross_val_score(nn, pca_X, Y, cv=n)
    scores_pca.append(np.mean(result))
    std_pca.append(np.std(result))

plt.errorbar(folds, scores_pca, yerr=std_pca, label='PCA')
plt.xlim((0, 60))
plt.legend(loc='best')
plt.ylabel("mean accuracy with std")
plt.xlabel("number of folds")
title = 'error bar plot'
plt.title(title)
plt.savefig('../pic/errorBarPCA.png')
plt.close()

autoencoder_config = pickle.load(open( "../data/autoencoder_config.pickle", "rb" ))
autoencoder_config = autoencoder_config[5]

myae = ae.AutoEncoder(
        layers=[
            ae.Layer(activation='Sigmoid', units=X.shape[1]),
            ae.Layer("Sigmoid", units=autoencoder_config['n_encoder'])],
        n_iter=1000)

myae.fit(X)

mymlp = mlp.Classifier(
        layers=[
            mlp.Layer("Sigmoid", units=X.shape[1]),
            mlp.Layer("Sigmoid", units=autoencoder_config['n_encoder']),
            mlp.Layer("Softmax")],
        n_iter=1000)
myae.transfer(mymlp)

scores_autoencoder = []
std_autoencoder = []
for n in folds:
    print('%d folds for AutoEncoder' % n)
    result = cross_val_score(mymlp, X, Y, cv=n)
    scores_autoencoder.append(np.mean(result))
    std_autoencoder.append(np.std(result))

plt.errorbar(folds, scores_autoencoder, yerr=std_autoencoder, label='AutoEncoder', ecolor='blue', color='blue')
plt.xlim((0, 60))
plt.legend(loc='best')
plt.ylabel("mean accuracy with std")
plt.xlabel("number of folds")
title = 'error bar plot'
plt.title(title)
plt.savefig('../pic/errorBarAutoEncoder.png')
plt.close()