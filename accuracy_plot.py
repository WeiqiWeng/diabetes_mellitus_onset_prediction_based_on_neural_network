import pickle
import matplotlib.pyplot as plt
plt.style.use('ggplot')


pca_config = pickle.load(open( "../data/pca_config.pickle", "rb" ))
autoencoder_config = pickle.load(open( "../data/autoencoder_config.pickle", "rb" ))

unit_list = range(1, 9)
pca_accuracy = []
autoencoder_accuracy = []

for x in pca_config:
    pca_accuracy.append(x['accuracy'])

for x in autoencoder_config:
    autoencoder_accuracy.append(x['accuracy'])

plt.plot(unit_list, pca_accuracy, label='PCA CV Mean accuracy')
for a,b in zip(unit_list, pca_accuracy):
    plt.text(a, b, str(round(b, 3)))
plt.plot(unit_list, autoencoder_accuracy, label='AutoEncoder CV Mean accuracy')
for a,b in zip(unit_list, autoencoder_accuracy):
    plt.text(a, b, str(round(b, 3)))
plt.legend(loc='best')
plt.ylabel("accuracy")
plt.xlabel("number of component/encoder unit")
title = 'accuracy plot'
plt.title(title)
plt.savefig('../pic/bestAccuracy.png')
plt.close()
