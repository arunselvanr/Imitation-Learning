# Package imports
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import os
import sklearn
import sklearn.datasets
import sklearn.linear_model

# number of sampling point
N = 500

# system model
F1 = np.array([[1.1193, 0.1978], [0.2916, 1.2877]])
G1 = np.array([[0.7482], [0.4505]])
F2 = np.array([[0.5108, 0.7948], [0.8176, 0.6443]])
G2 = np.array([[0.9133], [0.1524]])
K1 = np.array([[0.6900, 1.1817]])
K2 = np.array([[0.6638, 0.7314]])

# initial conditions
x1 = np.random.randn(2,1)
x2 = np.random.randn(2,1)
w1 = np.random.randn(2,1)
w2 = np.random.randn(2,1)

# creat cache
x1_n = []
x2_n = []
y    = []

for ind in range(N):
    x1_n.append(LA.norm(x1))
    x2_n.append(LA.norm(x2))
    if (x1_n[ind] ** 3 < 10 * x2_n[ind]):
        x1 = np.dot(F1, x1) + w1
        x2 = np.dot(F2 - G2*K2, x2) + w2
        y.append(0)
        plt.scatter(x1_n[ind], x2_n[ind], color = 'red')
    else:
        x1 = np.dot(F1 - G1*K1, x1) + w1
        x2 = np.dot(F2, x2) + w2
        y.append(1)
        plt.scatter(x1_n[ind], x2_n[ind], color = 'blue')
    w1 = np.random.randn(2,1)
    w2 = np.random.randn(2,1)

plt.show()

myfile = open("data3.txt", "a")

for idx in range(N):
    myfile.write("%f,%f,%f\n" %(x1_n[idx], x2_n[idx], y[idx]))
    myfile.write("\n")
    print(x1_n[idx], x2_n[idx], y[idx], myfile)

myfile.close()
#myfile = open("test.txt", "r")
#content = myfile.readlines()
#content = [x.strip('\n') for x in content]
#myfile.close()

#import os
#path = os.getcwd() + '/test.txt'
#data = pd.read_csv(path, header=None, names=['Plant 1', 'Plant 2', 'Schedule'])
