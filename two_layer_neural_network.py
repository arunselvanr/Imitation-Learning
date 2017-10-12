# Package imports
import numpy as np
import pandas as pd
import os
import sklearn
import sklearn.datasets
import sklearn.linear_model

def sigmoid(x):
    """
    """
    s = 1/(1 + np.exp(-x))

    return s

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape
    Y -- labels of shape

    Returns:
    n_x --
    n_h --
    n_y --
    """
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer

    return (n_x, n_h, n_y)

def initialize_parameters(n_x, n_h, n_y):
    """
    Arguments:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2) # we set up a seed so that your output matches ours
                      # although the initialization is random.

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)

    Returns:
    A2 -- The sigmoid output
    """
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Implement Forward Propagation to calculate A2 (probabilities)
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    assert(A2.shape == (1, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache

def compute_cost(A2, Y, parameters):

    m = Y.shape[1] # number of examples

    # Compute the cross-entropy cost
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    cost = - np.sum(logprobs)/m

    cost = np.squeeze(cost) # makes sure cost is the dimension we expect.

    assert(isinstance(cost, float))

    return cost

def backward_propagation(parameters, cache, X, Y):

    m = X.shape[1]

    # First, retrieve also W1 and W2 from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    # Retrieve also A1 and A2 from the dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward Propagation: calculate dW1, db1, dW2, db2.
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis = 1, keepdims = True)/m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis = 1, keepdims = True)/m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters, grads, learning_rate = 1.2):

    # Retrieve each parameter from the dictionary "parameters".
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads".
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def nn_model(X, Y, n_h, num_iterations = 10000, print_cost = False):

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2.
    # Inputs: "n_x, n_h, n_y".
    # Outputs: "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward Propagation.
        # Inputs: "X, parameters"
        # Outputs: "A2, cache"
        A2, cache = forward_propagation(X, parameters)

        # Cost function.
        # Inputs: "A2, Y, parameters"
        # Outputs: "cost"
        cost = compute_cost(A2, Y, parameters)

        # Backpropagation.
        # Inputs: "parameters, cache, X, Y"
        # Outputs: "grads"
        grads = backward_propagation(parameters, cache, X, Y)

        # Gradient descent parameter update.
        # Inputs: "parameters, grads"
        # Outputs: "parameters"
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

    return parameters

def predict(parameters, X):

    # Computes probabilities using forward propagation, and classifies to 0/1
    # using 0.5 as the threshold
    A2, cache = forward_propagation(X, parameters)
    predictions = (A2 > 0.5)

    return predictions

#
path = os.getcwd() + '/data/data3.txt'
data = pd.read_csv(path, header=None, names=['Plant 1', 'Plant 2', 'Schedule'])
data.head()

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1].T
Y = data.iloc[:,cols-1:cols].T

# convert to numpy arrays and initalize the parameter array theta
X = np.array(X.values)
Y = np.array(Y.values)

# Build a model a n_h-dimensional hidden layer
parameters = nn_model(X, Y, n_h = 5, num_iterations = 10001, print_cost = True)

# Print accuracy
predictions = predict(parameters, X)
print("Accuracy: {} %".format(float(np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100))

"""
N = len(predictions.T)

#
myfile = open("new_data3.txt", "a")

for idx in range(N):
    myfile.write("%f,%f,%f\n" %(X[0,idx], X[1,idx], predictions[0,idx]))
    myfile.write("\n")
    print(X[0,idx], X[1,idx], predictions[0,idx], myfile)

myfile.close()
"""
