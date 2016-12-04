# CS6140Projet

**Dependency**
* Python 3.4+
* Python module
	* matplotlib (1.5.3)
	* numpy (1.11.2)
	* pandas (0.19.1)
	* scikit-learn (0.18.1)
	* scipy (0.18.1)
	* six (1.10.0)
	* sklearn (0.0)
	* scikit-neuralnetwork (0.4)
	* Theano (0.8.2)
	* imbalanced-learn (0.1.8)
		
**Code structure**
* **data_manipulation.py:** apply combination of Synthetic Minority Over-sampling Technique (SMOTE) and Tomek Link to original data.
* **data_split.py:** split balanced dataset into _training set_, _validation set_ and _test set_ following 80-10-10 rule.
* **model_pca.py:** perform Principal Component Analysis upon the _training set_ while configuration for potential dimension reduction stored in _pca_config.pickle_
* **model_autoencoder.py:** similarily perform Auto-encoder to do dimension reduction.
* **model_performance.py:** build two neural-network with optimal configuration of PCA and Auto-encoder and evaluate prediction accuracy upon training set
* **model_cv.py:** run k-fold cross validation to calculate mean accuracy, standard deviation and report error bar plot for each architecture on the validation set
* **model_test.py:** build prediction model with optimal dimension reduction configuration and make prediction upon data sets
* **model_confusion.py:** generate confusion matrix with prediction model and calculate sensitivity and specificity
* **accuracy_plot.py:** make comparison between models generated with different dimension reduction configuration 
