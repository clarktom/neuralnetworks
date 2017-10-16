import time
import pprint
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


np.random.seed(10)

epochs = 1000
batch_size = 256
no_hidden1 = 60 #num of neurons in hidden layer 1
learning_rate = 0.0001

floatX = theano.config.floatX

# scale and normalize input data
def scale(X):
    X_max = np.max(X, axis=0)
    X_min = np.min(X, axis=0)
    return (X - X_min)/(X_max - X_min)
 
def normalize(X):
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    return (X - X_mean)/X_std

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

#read and divide data into test and train sets
cal_housing = np.loadtxt('../../data/cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

X_data, Y_data = shuffle_data(X_data, Y_data)

no_features = X_data.shape[1] 
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')

# initialize weights and biases for hidden layer(s) and output layer
w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX)
b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

w_h2 = theano.shared(np.random.randn(no_hidden1, 20)*.01, floatX)
b_h2 = theano.shared(np.random.randn(20)*0.01, floatX)

w_o = theano.shared(np.random.randn(20)*.01, floatX)
b_o = theano.shared(np.random.randn()*.01, floatX)

# learning rate
alpha = theano.shared(learning_rate, floatX)

#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
h2_out = T.nnet.sigmoid(T.dot(h1_out,w_h2) + b_h2)

y = T.dot(h2_out, w_o) + b_o # 4-layer

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

#define gradients
dw_o, db_o, dw_h1, db_h1, dw_h2, db_h2 = T.grad(cost, [w_o, b_o, w_h1, b_h1, w_h2, b_h2]) # 4-layer 

train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h1],
                   [b_h1, b_h1 - alpha*db_h1],
                   [w_h2, w_h2 - alpha*dw_h2],
                   [b_h2, b_h2 - alpha*db_h2]],
        allow_input_downcast=True
        ) # 4-layer

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )

min_error = 1e+15

noFolds = 5
print("X shape: ", X_data.shape)
print("Y shape: ", Y_data.shape)
fold_size = X_data.shape[0] // noFolds

alpha.set_value(learning_rate)
print(alpha.get_value())

fold_test_cost_min = []
fold_test_cost = []
fold_train_cost = []
fold_test_accuracy = []
for fold in range(noFolds):
    print("        Folder number: ", fold+1)

    start, end = fold*fold_size, (fold +1)*fold_size
    testX, testY = X_data[start:end], Y_data[start:end]
    trainX, trainY = np.append(X_data[:start], X_data[end:], axis=0), np.append(Y_data[:start], Y_data[end:], axis=0)

    trainX = normalize(trainX)
    testX = normalize(testX)

    print("            testX shape: ", testX.shape)
    print("            testY shape: ", testY.shape)

    w_h1.set_value(np.random.randn(no_features, no_hidden1)*.01)
    b_h1.set_value(np.random.randn(no_hidden1)*0.01)
    w_h2.set_value(np.random.randn(no_hidden1, 20)*.01)
    b_h2.set_value(np.random.randn(20)*0.01)
    w_o.set_value(np.random.randn(20)*.01)
    b_o.set_value(np.random.randn()*.01)

    min_cost = 1e+15
    min_accuracy = 1e+15
    epochs_test_cost_min = []
    epochs_test_cost = []
    epochs_train_cost = []
    epochs_test_accuracy = []
    for epoch in range(epochs):
        n = trainX.shape[0]
        train_cost = 0
        for start_batch, end_batch in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            train_cost += train(trainX[start_batch:end_batch], np.transpose(trainY[start_batch:end_batch]))
        pred, test_cost, test_accuracy = test(testX, np.transpose(testY))
        epochs_test_cost.append(test_cost)
        epochs_test_accuracy.append(test_accuracy)
        epochs_train_cost.append(train_cost/(n // batch_size))

        if test_cost < min_cost:
            min_cost = test_cost
        if test_accuracy < min_accuracy:
            min_accuracy = test_accuracy

    fold_test_cost_min.append(min_cost)
    fold_test_cost.append(epochs_test_cost)
    fold_train_cost.append(epochs_train_cost)
    fold_test_accuracy.append(epochs_test_accuracy)

fold_test_cost = np.mean(fold_test_cost, axis=0)
print("fold_test_cost")
pprint.pprint(fold_test_cost)
fold_train_cost = np.mean(fold_train_cost, axis=0)
print("fold_train_cost")
pprint.pprint(fold_train_cost)
fold_test_accuracy = np.mean(fold_test_accuracy, axis=0)
print("fold_train_cost")
pprint.pprint(fold_test_accuracy)

plt.figure()
plt.plot(range(epochs), fold_test_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy for 4 layers')
plt.savefig('p_1b_sample_accuracy.png')
plt.show()
