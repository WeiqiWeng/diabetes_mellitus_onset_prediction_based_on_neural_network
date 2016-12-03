import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

def compute(nn, trn_X, trn_Y, tst_X, tst_Y):
    nn.fit(trn_X, trn_Y)
    result = nn.predict(tst_X)

    confusion = [[0, 0], [0, 0]]
    for i in range(len(tst_Y)):
        if (tst_Y[i] > 0.0 and result[i] > 0.0):
            confusion[1][1] += 1
        elif (tst_Y[i] > 0.0 and result[i] < 1.0):
            confusion[1][0] += 1
        elif (tst_Y[i] < 1.0 and result[i] > 0.0):
            confusion[0][1] += 1
        elif (tst_Y[i] < 1.0 and result[i] < 1.0):
            confusion[0][0] += 1

    return confusion[1][1] / len(tst_Y), confusion[0][0] / len(tst_Y)

# Train model (PCA)
df = pd.read_pickle('../data/diabetes_train.pickle')
df_validation = pd.read_pickle('../data/diabetes_validation.pickle')
df_train = np.array(pd.concat([df, df_validation]))

X = df_train[:, 0:-1]
trn_Y = df_train[:, -1]

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


testFile = np.array(pd.read_pickle('../data/diabetes_test.pickle'))
X = testFile[:, 0:-1]
tst_Y = testFile[:, -1]

tst_X = pca.transform(X)
sen_list = []
spec_list = []
i_list = []
maxi = 100
for i in range(maxi - 5):
    mean_sen, mean_spec = 0.0, 0.0
    print(i)
    i_list.append(i + 5)
    for j in range(i + 5):
        sen, spec = compute(nn, pca_X, trn_Y, tst_X, tst_Y)
        mean_sen += sen
        mean_spec += spec
    sen_list.append(mean_sen / (i + 5))
    spec_list.append(mean_spec / (i + 5))


plt.style.use('ggplot')
plt.xlim(0, maxi)
plt.plot(i_list, sen_list, 'ro')
plt.plot(i_list, spec_list, 'bo')
plt.ylim(0.2, 0.5)
title = 'Mean Sensitivity (r) and specificity (b)'
plt.title(title)
plt.xlabel('epochs')
plt.ylabel('mean')
plt.legend(loc='best')
plt.savefig('../pic/meanSens&Spec.png')
plt.close()