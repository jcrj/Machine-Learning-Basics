from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#SVM is a classifier, can sometimes be more effective than a simple neural network
#Find the most generalised and optimal line which separate the data points into 2 distinct sets -> then generate the 2 support vectors parallel to the former line
#with gaps in between the support vectors and the the optimal line
#With real life data, sometimes it is very difficult to find distinct classifiers (more convoluted)
#We then use kernels to add another dimension to better separate such data points
#Soft margins -> tolerance for mis-classification

data = load_breast_cancer()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

model = SVC(kernel = 'linear', C = 3)
model.fit(x_train, y_train)

model2 = KNeighborsClassifier(n_neighbors = 3)
model2.fit(x_train, y_train)

print(f'SVC: {model.score(x_test, y_test)}') #svm
print(f'KNN: {model2.score(x_test, y_test)}' ) #knn

#In this case, SVM outperforms KNN (0.953 > 0.918)