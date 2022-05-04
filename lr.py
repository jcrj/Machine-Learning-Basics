from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

#f(x) = m*x + b
#finding the best fit line that minimises(optimising) the least squares distance between the data points and the fitted line

#random time to scores in exam (self-generated)
time = np.array([30,45,60,33,69,20,22,90,85]).reshape(-1,1) 
score = np.array([55,80,35,32,57,68,13,95,78]).reshape(-1,1)
#sklearn model can only work when params are reshaped to (-1,1)

time_train, time_test, score_train, score_test = train_test_split(time,score, test_size = 0.2)

model = LinearRegression()
model.fit(time, score) #model.fit(x param, y param)

print(model.score(time, score))

plt.scatter(time_train, score_train)
plt.plot(np.linspace(0, 100, 5).reshape(-1,1), model.predict(np.linspace(0,100,5).reshape(-1,1)), 'y') #np.linspace(start, stop, num = xxx, endpoint = True, retstep = False, dtype = None, axis = 0)
plt.show()
#0.341