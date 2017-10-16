import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


def init_bias(n = 1):
    return(theano.shared(np.zeros(n), theano.config.floatX))

def init_weights(n_in=1, n_out=1, logistic=True):
    W_values = np.asarray(
        np.random.uniform(
        low=-np.sqrt(6. / (n_in + n_out)),
        high=np.sqrt(6. / (n_in + n_out)),
        size=(n_in, n_out)),
        dtype=theano.config.floatX
        )
    if logistic == True:
        W_values *= 4
    return (theano.shared(value=W_values, name='W', borrow=True))

# scale data
def scale(X, X_min, X_max): # TODO: Implement the other method
    return (X - X_min)/(X_max-np.min(X, axis=0))

def scaleN(X):
    return (X - np.mean(X))/np.std(X)

# update parameters
def sgd(cost, params, lr=0.01):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - g * lr])
    return updates

def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    #print  (samples.shape, labels.shape)
    samples, labels = samples[idx], labels[idx]
    return samples, labels


decay = 1e-6
learning_rate = 0.01
epochs = 1000
batch_size = 32

# theano expressions
X = T.matrix() #features
Y = T.matrix() #output

w1, b1 = init_weights(36, 10), init_bias(10) #weights and biases from input to hidden layer
w2, b2 = init_weights(10, 6, logistic=False), init_bias(6) #weights and biases from hidden to output layer

h1 = T.nnet.sigmoid(T.dot(X, w1) + b1)
py = T.nnet.softmax(T.dot(h1, w2) + b2)

y_x = T.argmax(py, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py, Y)) + decay*(T.sum(T.sqr(w1)+T.sum(T.sqr(w2))))
params = [w1, b1, w2, b2]
updates = sgd(cost, params, learning_rate)

# compile
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True) # TODO: Understand how this works
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

list1 = []
list2 = []
max_it = 20
for j in range(0, max_it):

    #read train data
    train_input = np.loadtxt('../../data/sat_train.txt',delimiter=' ')
    trainX, train_Y = train_input[:,:36], train_input[:,-1].astype(int)
    trainX_min, trainX_max = np.min(trainX, axis=0), np.max(trainX, axis=0)

    print(j)
    # print(j < max_it/2)
    if j < max_it/2:
        trainX = scale(trainX, trainX_min, trainX_max)
    else:
        trainX = scaleN(trainX)

    train_Y[train_Y == 7] = 6
    trainY = np.zeros((train_Y.shape[0], 6))
    trainY[np.arange(train_Y.shape[0]), train_Y-1] = 1 # That's K matrix

    #read test data
    test_input = np.loadtxt('../../data/sat_test.txt',delimiter=' ')
    testX, test_Y = test_input[:,:36], test_input[:,-1].astype(int)

    testX_min, testX_max = np.min(testX, axis=0), np.max(testX, axis=0)
    if j < max_it/2:
        testX = scale(testX, testX_min, testX_max)
    else:
        testX = scaleN(testX)

    test_Y[test_Y == 7] = 6
    testY = np.zeros((test_Y.shape[0], 6))
    testY[np.arange(test_Y.shape[0]), test_Y-1] = 1


    # train and test
    n = len(trainX)
    test_accuracy = []
    train_cost = []
    for i in range(epochs):

        trainX, trainY = shuffle_data(trainX, trainY)
        cost = 0.0
        for start, end in zip(range(0, n, batch_size), range(batch_size, n, batch_size)):
            cost += train(trainX[start:end], trainY[start:end])
        train_cost = np.append(train_cost, cost/(n // batch_size))

        test_accuracy = np.append(test_accuracy, np.mean(np.argmax(testY, axis=1) == predict(testX)))

    if j < max_it/2:
        list1.append(test_accuracy)
    else:
        list2.append(test_accuracy)


print("Computing means...")
list1 = np.mean(list1, axis=0)
list2 = np.mean(list2, axis=0)

print("Plotting results...")
plt.figure()
plt.plot(range(epochs), list1, label="Scale eq 1")
plt.plot(range(epochs), list2, label="Scale eq 2")
plt.legend(loc="bottom right")
plt.xlabel('iterations')
plt.ylabel('accuracy')
plt.title('test accuracy')
plt.savefig('p1a_sample_accuracy.png')

plt.show()
