from sklearn.datasets import load_breast_cancer #datasets
from sklearn.neighbors import KNeighborsClassifier #knn from sklearn package
from sklearn.model_selection import train_test_split #sklearn package to split data into training and testing datasets
import numpy as np

data = load_breast_cancer()

#print(data.feature_names) #features of breast cancer
#print(data.target_names) #malignant, benign

x_train, x_test, y_train, y_test = train_test_split(np.array(data.data), np.array(data.target), test_size = 0.3) #taking 30% of the data into training data and another 30% into testing data (randomising)

knn_classifier = KNeighborsClassifier(n_neighbors = 3) #malignant + benign = 2, +1 to have 3 total classifiers to take care of edge cases where data point is equidistant to malignant and benign
knn_classifier.fit(x_train, y_train)

print(knn_classifier.score(x_test, y_test)) #testing how well a model performs
#0.9298245614035088
#Correct 9/10 times in classifying breast cancer tumors