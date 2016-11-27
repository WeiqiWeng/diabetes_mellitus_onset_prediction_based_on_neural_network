import pandas as pd
import numpy as np
import pickle
from sknn.backend import pylearn2
from sklearn.model_selection import GridSearchCV
from sknn import ae, mlp
import matplotlib.pyplot as plt
plt.style.use('ggplot')


df = np.array(pd.read_pickle('../data/diabetes_train.pickle'))

X = df[:, 0:-1]
Y = df[:, -1]

score = -1
opt_n_encoder = 0
opt_params = 0
config = []

layer1 = ae.Layer(activation='Sigmoid', units=X.shape[1])
param_grid = {'weight_decay': np.arange(0.0, 1.01, 0.01)}

unit_list = range(1, 9)
for n in unit_list:
    print('grid search for n = %d' % n)
    myae = ae.AutoEncoder(
        layers=[
            layer1,
            ae.Layer("Sigmoid", units=n)],
        n_iter=1000)

    myae.fit(X)

    mymlp = mlp.Classifier(
        layers=[
            mlp.Layer("Sigmoid", units=X.shape[1]),
            mlp.Layer("Sigmoid", units=n),
            mlp.Layer("Softmax")],
        n_iter=1000)
    myae.transfer(mymlp)
    gs = GridSearchCV(mymlp, param_grid)
    gs.fit(X, Y)

    config.append({'n_encoder': n, 'accuracy': gs.best_score_, 'param': gs.best_params_})
    if gs.best_score_ > score:
        opt_n_encoder = n
        opt_params = gs.best_params_
        score = gs.best_score_

pickle.dump(config, open("../data/autoencoder_config.pickle", "wb"))

print('optimal number of encoder units: ', opt_n_encoder)
print('optimal weight decay: ', opt_params['weight_decay'])
print('accuracy after autoencoder: ', score)