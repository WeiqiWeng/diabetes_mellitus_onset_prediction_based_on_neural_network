import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
plt.style.use('ggplot')


df = np.array(pd.read_pickle('../data/diabetes_normalized_balanced.pickle'))
X = df[:, 0:-1]
Y = df[:, -1]

pca_config = pickle.load(open( "../data/pca_config.pickle", "rb" ))
pca_config = pca_config[5]

epoch_list = range(5, 1001)

n_component = pca_config['n_components']
hidden_layer_size = pca_config['param']['hidden_layer_sizes']
weight_decay = pca_config['param']['alpha']

pca = PCA(n_components=n_component)
pca_X = pca.fit_transform(X)

nn = MLPClassifier(solver='lbfgs', max_iter=1000, alpha=weight_decay, hidden_layer_sizes=(hidden_layer_size))

mean_accuracy_list = []
for epoch in epoch_list:
    mean_accuracy = 0.0
    for round in range(epoch):
        print('epoch = %d, round %d' % (epoch, round))
        nn.fit(pca_X, Y)
        mean_accuracy += nn.score(pca_X, Y)

    mean_accuracy_list.append(mean_accuracy/epoch)

plt.scatter(epoch_list, mean_accuracy_list, label='mean accuracy', marker='.', linewidths=0.3)
plt.xlim((0, 1200))
plt.ylim((0, 1.2))
plt.legend(loc='best')
plt.ylabel("mean accuracy")
plt.xlabel("epochs")
title = 'mean accuracy'
plt.title(title)
plt.savefig('../pic/meanAccuracy.png')
plt.close()
