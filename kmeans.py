from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
from sklearn.datasets import load_digits

#unsupervised-learning method
#find k amount of clusters based on random centroids generated
#after the initial classification, place new centroids in the middle of its assigned cluster and start the process again
#when centroids are not moving significantly anymore, terminate and return

digits = load_digits()
data = scale(digits.data)

model = KMeans(n_clusters = 10, init = 'random', n_init = 10) #10 digits, placement of centroids is random, initialise 10 centroids
model.fit(data)

