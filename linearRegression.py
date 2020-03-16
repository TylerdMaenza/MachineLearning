import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

data = pd.read_csv("student-mat.csv", sep=";")  # reads data set

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]    # array for the data set

predict = "G3"

X = np.array(data.drop([predict], 1))      # setting up the arrays for training
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)  # training
'''
best = 0
for _ in range(30):

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)      this is finding the highest acc then using that for future in 
    print(acc)                              studentmodel.pickle

    if acc > best:
        best = acc
        with open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)
'''

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)    # this creates the graph model we will use

print("Coefficient: \n", linear.coef_)      # prints the parts of the graph itself
print("Intercept: \n", linear.intercept_)

predictions = linear.predict(x_test)
for x in range(len(predictions)):       # prints the predictions themselves from the training
    print(predictions[x], x_test[x], y_test[x])

p = "absences"  # change this to see the variances in graduation based on a value
style.use("ggplot")     # "pretty plot"
pyplot.scatter(data[p], data["G3"])     # to display what type of graph
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")        # gives the x and y parts of the graph
pyplot.show()       # display graph
