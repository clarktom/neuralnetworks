import time
import numpy as np
import theano
import theano.tensor as T

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn.model_selection import KFold



np.random.seed(10)

epochs = 1000
batch_size = 128
no_hidden1 = 30 #num of neurons in hidden layer 1
learning_rate = 0.0001
noExps = 2

floatX = theano.config.floatX

# scale and normalize input data
def scale(X, X_min, X_max):
    X_max, X_min =  np.max(X, axis=0), np.min(X, axis=0)
    return (X - X_min)/(X_max - X_min)
 
def normalize(X, X_mean, X_std):
    X_mean, X_std = np.mean(X, axis=0), np.std(X, axis=0)
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

# X_data, Y_data = shuffle_data(X_data, Y_data)

#separate train and test data
# m = 3*X_data.shape[0] // 10
# testX, testY = X_data[:m],Y_data[:m]
# trainX, trainY = X_data[m:], Y_data[m:]

# trainX = scale(trainX)
# testX = scale(testX)

# trainX = normalize(trainX, trainX_mean, trainX_std)
# testX = normalize(testX, testX_mean, testX_std)

no_features = X_data.shape[1] 
x = T.matrix('x') # data sample
d = T.matrix('d') # desired output
no_samples = T.scalar('no_samples')

# initialize weights and biases for hidden layer(s) and output layer
# w_o = theano.shared(np.random.randn(no_hidden1)*.01, floatX ) 
# b_o = theano.shared(np.random.randn()*.01, floatX)
# w_h1 = theano.shared(np.random.randn(no_features, no_hidden1)*.01, floatX )
# b_h1 = theano.shared(np.random.randn(no_hidden1)*0.01, floatX)

w_h1, b_h1 = init_weights(no_features, no_hidden1), init_bias(no_hidden1)
w_o, b_o = init_weights(no_hidden1), init_bias()

# learning rate
alpha = theano.shared(learning_rate, floatX)

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


train_cost = np.zeros(epochs)
test_cost = np.zeros(epochs)
test_accuracy = np.zeros(epochs)

min_error = 1e+15
# best_iter = 0
# best_w_o = np.zeros(no_hidden1)
# best_w_h1 = np.zeros([no_features, no_hidden1])
# best_b_o = 0
# best_b_h1 = np.zeros(no_hidden1)

alpha.set_value(learning_rate)
print(alpha.get_value())

# t = time.time()
# for iter in range(epochs):
#     if iter % 100 == 0:
#         print(iter)
    
#     trainX, trainY = shuffle_data(trainX, trainY)
#     train_cost[iter] = train(trainX, np.transpose(trainY))
#     pred, test_cost[iter], test_accuracy[iter] = test(testX, np.transpose(testY))

#     if test_cost[iter] < min_error:
#         best_iter = iter
#         min_error = test_cost[iter]
#         best_w_o = w_o.get_value()
#         best_w_h1 = w_h1.get_value()
#         best_b_o = b_o.get_value()
#         best_b_h1 = b_h1.get_value()

learning_rates = [0.001, 0.005, 0.0001, 0.0005, 0.00001]

noFolds = 5
print("X shape: ", X_data.shape)
fold_size = X_data.shape[0] // noFolds
print("Fold size: ", fold_size)
print("nb of features: ", no_features)


opt_learningrate = []
for exp in range(noExps):
    print('Exp: ', exp+1)
    # np.random.shuffle(idx)
    # X, Y = X[idx], Y[idx]
    X_data, Y_data = shuffle_data(X_data, Y_data)
    
    test_cost = []
    for learning_rate in learning_rates:
        print("Learning rate: ", learning_rate)
        alpha.set_value(learning_rate)

        fold_cost = []
        for fold in range(noFolds):

            start, end = fold*fold_size, (fold +1)*fold_size
            testX, testY = X_data[start:end], Y_data[start:end]
            trainX, trainY = np.append(X_data[:start], X_data[end:], axis=0), np.append(Y_data[:start], Y_data[end:], axis=0)
        
            set_weights(w_h1, no_features, no_hidden1)
            set_bias(b_h1, no_hidden1)
            set_weights(w_o, no_hidden1)
            set_bias(b_o)

            min_cost = 100000
            # train_cost = []
            for epoch in range(epochs):
                n = trainX.shape[0]
                for start_batch, end_batch in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
                    train(trainX[start_batch:end_batch], trainY[start_batch:end_batch])
                # train_cost.append(cost/(n // batch_size))
                pred, test_cost, test_accuracy = test(testX, testY)
                if test_cost < min_cost:
                    min_cost = test_cost
            fold_cost = np.append(fold_cost, min_cost)

        # print("Fold", fold_cost.shape)
        test_cost = np.append(test_cost, np.mean(fold_cost))
        # print("Test", test_cost.shape)
    opt_learningrate = np.append(opt_learningrate, np.argmin(test_cost)+1)


print(opt_learningrate)
plt.figure()
counts = []
for learning_rate in learning_rates:
    counts = np.append(counts, np.sum(opt_learningrate == learning_rate))
plt.bar(np.arange(1, len(learning_rates)+1), counts)
plt.xticks(np.arange(0, len(learning_rates)+1, len(learning_rates) // 4))
plt.xlabel('number of hidden neurons')
plt.ylabel('number of experiments')
plt.title('distribution of optimal number of hidden neurons')
plt.savefig('figure_t6.q1b_2.png')
plt.show()

#set weights and biases to values at which performance was best
# w_o.set_value(best_w_o)
# b_o.set_value(best_b_o)
# w_h1.set_value(best_w_h1)
# b_h1.set_value(best_b_h1)
    
# best_pred, best_cost, best_accuracy = test(testX, np.transpose(testY))

# print('Minimum error: %.1f, Best accuracy %.1f, Number of Iterations: %d'%(best_cost, best_accuracy, best_iter))

#Plots
plt.figure()
plt.plot(range(epochs), train_cost, label='train error')
plt.plot(range(epochs), test_cost, label = 'test error')
plt.xlabel('epochs')
plt.ylabel('Mean Squared Error')
plt.title('Training and Test Errors at Alpha = %.4f'%learning_rate)
plt.legend()
plt.savefig('p_1b_sample_mse.png')
plt.show()

plt.figure()
plt.plot(range(epochs), test_accuracy)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.savefig('p_1b_sample_accuracy.png')
plt.show()