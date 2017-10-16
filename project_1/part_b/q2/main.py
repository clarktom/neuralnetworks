import pprint
import time
import datetime
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import KFold

np.random.seed(10)

epochs = 1000
batch_size = 256
no_hidden1 = 30
learning_rates = [0.001, 0.005, 0.0001, 0.0005, 0.00001]
learning_rate = 0.001
noExps = 10

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

def init_bias(n = 1):
    return(theano.shared(np.zeros(n), theano.config.floatX))

def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out))
    if logistic == True:
        W_values *= 4
    return(theano.shared(W_values, theano.config.floatX))

def set_bias(b, n = 1):
    b.set_value(np.zeros(n))

def set_weights(w, n_in=1, n_out=1, logistic=True):
    W_values = np.random.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                 high=np.sqrt(6. / (n_in + n_out)),
                                 size=(n_in, n_out))
    if logistic == True:
        W_values *= 4
    w.set_value(W_values)

#read and divide data into test and train sets 
cal_housing = np.loadtxt('../../data/cal_housing.data', delimiter=',')
X_data, Y_data = cal_housing[:,:8], cal_housing[:,-1]
Y_data = (np.asmatrix(Y_data)).transpose()

no_features = X_data.shape[1] 
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')

# # learning rate
alpha = theano.shared(learning_rate, floatX)

# initialize weights and biases for hidden layer(s) and output layer
w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX ) 
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

#Define mathematical expression:
h1_out = T.nnet.sigmoid(T.dot(x, w_h1) + b_h1)
y = T.dot(h1_out, w_o) + b_o

cost = T.abs_(T.mean(T.sqr(d - y)))
accuracy = T.mean(d - y)

#define gradients
dw_o, db_o, dw_h, db_h = T.grad(cost, [w_o, b_o, w_h1, b_h1])

train = theano.function(
        inputs = [x, d],
        outputs = cost,
        updates = [[w_o, w_o - alpha*dw_o],
                   [b_o, b_o - alpha*db_o],
                   [w_h1, w_h1 - alpha*dw_h],
                   [b_h1, b_h1 - alpha*db_h]],
        allow_input_downcast=True
        )

test = theano.function(
    inputs = [x, d],
    outputs = [y, cost, accuracy],
    allow_input_downcast=True
    )

min_error = 1e+15

alpha.set_value(learning_rate)
print(alpha.get_value())

noFolds = 5
print("X shape: ", X_data.shape)
print("Y shape: ", Y_data.shape)
fold_size = X_data.shape[0] // noFolds
print("Fold size: ", fold_size)
print("nb of features: ", no_features)

print("---------------------")

opt_learningrate = []
exp_test_cost_min = []
exp_test_cost = []
exp_train_cost = []
for exp in range(noExps):
    start_time = time.time()
    print(datetime.datetime.now().time(), '- Exp: ', exp+1)
    X_data, Y_data = shuffle_data(X_data, Y_data)
    
    param_test_cost_min = []
    param_test_cost = []
    param_train_cost = []
    for learning_rate in learning_rates:
        print("    Learning rate: ", learning_rate)
        alpha.set_value(learning_rate)

        fold_test_cost_min = []
        fold_test_cost = []
        fold_train_cost = []
        for fold in range(noFolds):
            print("        Folder number: ", fold+1)

            start, end = fold*fold_size, (fold +1)*fold_size
            testX, testY = X_data[start:end], Y_data[start:end]
            trainX, trainY = np.append(X_data[:start], X_data[end:], axis=0), np.append(Y_data[:start], Y_data[end:], axis=0)

            trainX = normalize(trainX)
            testX = normalize(testX)

            print("            testX shape: ", testX.shape)
            print("            testY shape: ", testY.shape)

            w_o.set_value(np.random.randn(no_hidden1)*.01)
            b_o.set_value(np.random.randn()*.01)
            w_h1.set_value(np.random.randn(no_features, no_hidden1)*.01)
            b_h1.set_value(np.random.randn(no_hidden1)*0.01)

            min_cost = 1e+15
            min_accuracy = 1e+15
            epochs_test_cost_min = []
            epochs_test_cost = []
            epochs_train_cost = []
            for epoch in range(epochs):
                n = trainX.shape[0]
                train_cost = 0
                for start_batch, end_batch in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
                    train_cost += train(trainX[start_batch:end_batch], np.transpose(trainY[start_batch:end_batch]))
                pred, test_cost, test_accuracy = test(testX, np.transpose(testY))
                epochs_test_cost.append(test_cost)
                epochs_train_cost.append(train_cost/(n // batch_size))

                if test_cost < min_cost:
                    min_cost = test_cost
                if test_accuracy < min_accuracy:
                    min_accuracy = test_accuracy

            fold_test_cost_min.append(min_cost)
            fold_test_cost.append(epochs_test_cost)
            fold_train_cost.append(epochs_train_cost)

        param_test_cost_min.append(np.mean(fold_test_cost_min))
        param_test_cost.append(np.mean(fold_test_cost, axis=0))
        param_train_cost.append(np.mean(fold_train_cost, axis=0))

    exp_test_cost.append(param_test_cost)
    exp_train_cost.append(param_train_cost)

    opt_learningrate = np.append(opt_learningrate, np.argmin(param_test_cost_min))
    elapsed_time = time.time() - start_time
    print("    Elapsed time:", elapsed_time)

print("opt_learningrate: ", opt_learningrate)

exp_test_cost = np.mean(exp_test_cost, axis=0)
print("exp_test_cost")
pprint.pprint(exp_test_cost)
exp_train_cost = np.mean(exp_train_cost, axis=0)
print("exp_train_cost")
pprint.pprint(exp_train_cost)

plt.figure()
counts = []
for n in range(len(learning_rates)):
    counts = np.append(counts, np.sum(opt_learningrate == n))
print("counts: ", counts)
plt.bar(np.arange(1, len(learning_rates)+1), counts)
plt.xticks(np.arange(1, len(learning_rates)+1), learning_rates)
plt.xlabel('learning rates')
plt.ylabel('number of experiments')
plt.title('distribution of optimal learning rate')
plt.savefig('figure_t6.q1b_2.png')
plt.show()

#Plots
plt.figure()
for idx, learning_rate in enumerate(learning_rates):
    print("exp_train_cost", idx)
    pprint.pprint(exp_train_cost[idx])
    print("exp_test_cost", idx)
    pprint.pprint(exp_test_cost[idx])
    plt.plot(range(epochs), exp_train_cost[idx], label=('train error '+str(learning_rate)))
    plt.plot(range(epochs), exp_test_cost[idx], label=('validation error '+str(learning_rate)))
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training and Validation Errors')
plt.legend()
plt.savefig('p_1b_sample_mse.png')
plt.show()

# --------------------------------------------------------------------------------------------------------

noFolds = 5
print("X shape: ", X_data.shape)
print("Y shape: ", Y_data.shape)
fold_size = X_data.shape[0] // noFolds

w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX ) 
b_o = theano.shared(np.random.randn()*.01, floatX)
w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

best_learning_rate = 0.00001
alpha.set_value(best_learning_rate)
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

    w_o.set_value(np.random.randn(no_hidden1)*.01)
    b_o.set_value(np.random.randn()*.01)
    w_h1.set_value(np.random.randn(no_features, no_hidden1)*.01)
    b_h1.set_value(np.random.randn(no_hidden1)*0.01)

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

#Plots
plt.figure()
plt.plot(range(epochs), fold_train_cost, label='train error')
plt.plot(range(epochs), fold_test_cost, label = 'test error')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors at Alpha = %.5f'%best_learning_rate)
plt.legend()
plt.savefig('p_1b_sample_mse.png')
plt.show()

plt.figure()
plt.plot(range(epochs), fold_test_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.savefig('p_1b_sample_accuracy.png')
plt.show()