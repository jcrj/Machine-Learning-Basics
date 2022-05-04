from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

#Basically yes/no and branch.
#Find a root node and split up into possibilities -> branch yes/no to next node and continue
#Based on the factor we are weighing, judge based on probability and branch

#Random forest classifications -> generating multiple decision trees and feeding the same input and use the collective results to generate the final result -> minimise
#the risk of mis-classification

data = load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

model = SVC(kernel = 'linear', C = 3)
model.fit(x_train, y_train)

model2 = KNeighborsClassifier(n_neighbors = 3)
model2.fit(x_train, y_train)

model3 = DecisionTreeClassifier()
model3.fit(x_train, y_train)

model4 = RandomForestClassifier()
model4.fit(x_train, y_train)

print(f'SVC: {model.score(x_test, y_test)}') #svm
print(f'KNN: {model2.score(x_test, y_test)}' ) #knn
print(f'Decision Tree: {model.score(x_test, y_test)}') #decision tree
print(f'Random Forest: {model2.score(x_test, y_test)}' ) #random forest