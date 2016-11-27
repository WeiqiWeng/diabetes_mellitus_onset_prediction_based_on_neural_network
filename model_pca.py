import pandas as pd
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
plt.style.use('ggplot')

df = np.array(pd.read_pickle('../data/diabetes_train.pickle'))

X = df[:, 0:-1]
Y = df[:, -1]

score = -1
opt_n_components = 0
opt_X = 0
opt_params = dict()
config = []
coponent_list = range(1, 9)

hidden_layer_list = [(x,) for x in range(5, 21)]

for n in coponent_list:
    print('grid search for n = %d' % n)
    pca = PCA(n_components=n)
    pca_X = pca.fit_transform(X)

    nn = MLPClassifier(solver='lbfgs', max_iter=1000)
    param_grid = {
        'hidden_layer_sizes': hidden_layer_list,
        'alpha': np.arange(0.0, 1.01, 0.01)}
    gs = GridSearchCV(nn, param_grid)

    gs.fit(pca_X, Y)
    config.append({'n_components':n, 'pca_X':pca_X, 'accuracy':gs.best_score_, 'param':gs.best_params_})
    if gs.best_score_ > score:
        opt_n_components = n
        opt_X = pca_X
        opt_params = gs.best_params_
        score = gs.best_score_

pickle.dump(config, open("../data/pca_config.pickle", "wb"))

print('optimal number of components: ', opt_n_components)
print('optimal hidden layer size: ', opt_params['hidden_layer_sizes'])
print('optimal weight decay: ', opt_params['alpha'])
nn = MLPClassifier(solver='lbfgs',
                   max_iter=1000,
                   hidden_layer_sizes=opt_params['hidden_layer_sizes'],
                   alpha=opt_params['alpha'])
nn.fit(opt_X, Y)
print('accuracy after PCA: ', nn.score(opt_X, Y))

