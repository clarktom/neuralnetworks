#
# Tutorial 6, Question 1b: Cross-validation
#

import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

noHiddens = 10
noFolds = 5
noExps = 25
noIters = 20000

noInputs = 2
noOutputs = 1

alpha = 0.05
np.random.seed(10)

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

    
#Define variables:
x = T.matrix('x')
d = T.matrix('d')
w_h = init_weights(noInputs, noHiddens)
w_o = init_weights(noHiddens, noOutputs)
b_h = init_bias(noHiddens)
b_o = init_bias(noOutputs)


#Define mathematical expression:
z = T.nnet.sigmoid(T.dot(x, w_h)+b_h)
y = 2*T.nnet.sigmoid(T.dot(z, w_o)+b_o)-1
cost = T.mean(T.sum(T.sqr(d - y), axis=1))
dw_h, dw_o, db_h, db_o = T.grad(cost, [w_h, w_o, b_h, b_o])

# Compile
train = theano.function(
        inputs = [x, d],
        updates = [[w_o, w_o - alpha*dw_o],
                   [w_h, w_h - alpha*dw_h],
                   [b_o, b_o - alpha*db_o],
                   [b_h, b_h - alpha*db_h]],allow_input_downcast=True)

test = theano.function(inputs = [x, d],outputs = cost, allow_input_downcast=True)
predict = theano.function(inputs = [x],outputs = y, allow_input_downcast=True)


# generate training data
X = np.zeros((10*10, noInputs))
p = 0
for i in np.arange(-1, 1.001, 2/9):
    for j in np.arange(-1, 1.001, 2/9):
        X[p] = [i, j]
        p += 1
Y = np.zeros((p, 1))
Y[:,0] = np.sin(np.pi*X[:,0])*np.cos(2*np.pi*X[:,1])

idx = np.arange(100)
np.random.shuffle(idx)
X, Y = X[idx], Y[idx]

plt.figure()
plt.plot(X[:20,0], X[:20,1], 'rx', label='fold 1')
plt.plot(X[20:40,0], X[20:40,1], 'bx', label='fold 2')
plt.plot(X[40:60,0], X[40:60,1], 'gx', label='fold 3')
plt.plot(X[60:80,0], X[60:80,1], 'yx', label='fold 4')
plt.plot(X[80:,0], X[80:,1], 'mx', label='fold 5')
plt.legend()
plt.title('Training and test inputs')
plt.savefig('figure_t6.q1b_1.png')

# model selection for the number of hidden neurons
opt_hidden = []
for exp in range(noExps):
    print('exp:',exp)
    np.random.shuffle(idx)
    X, Y = X[idx], Y[idx]
    
    test_cost = []
    for hidden in np.arange(1, noHiddens+1):
        w_h = init_weights(noInputs, hidden)
        w_o = init_weights(hidden, noOutputs)
        b_h = init_bias(hidden)
        b_o = init_bias(noOutputs)

        fold_cost = []
        for fold in range(noFolds):
            start, end = fold*20, (fold +1)*20
            testX, testY = X[start:end], Y[start:end]
            trainX, trainY = np.append(X[:start], X[end:], axis=0), np.append(Y[:start], Y[end:], axis=0)
        
            set_weights(w_h, noInputs, hidden)
            set_weights(w_o, hidden, noOutputs)
            set_bias(b_h, hidden)
            set_bias(b_o, noOutputs)

            min_cost = 100000
            for iter in range(noIters):
                train(trainX, trainY)
                err = test(testX, testY)
                if err < min_cost:
                    min_cost = err
            fold_cost = np.append(fold_cost, min_cost)

        test_cost = np.append(test_cost, np.mean(fold_cost))
    opt_hidden = np.append(opt_hidden, np.argmin(test_cost)+1)

plt.figure()
counts = []
for hidden in np.arange(1, noHiddens+1):
    counts = np.append(counts, np.sum(opt_hidden == hidden))
plt.bar(np.arange(1, noHiddens+1), counts)
plt.xticks(np.arange(0, noHiddens+1, noHiddens // 4))
plt.xlabel('number of hidden neurons')
plt.ylabel('number of experiments')
plt.title('distribution of optimal number of hidden neurons')
plt.savefig('figure_t6.q1b_2.png')

# Plot the approximation
opt_hidden = np.argmax(counts)+1
print('optimal number of neurons is %d'%opt_hidden)

w_h = init_weights(noInputs, hidden)
w_o = init_weights(hidden, noOutputs)
b_h = init_bias(hidden)
b_o = init_bias(noOutputs)

min_cost = 10000
test_cost = []
for iter in range(noIters):
    train(X, Y)
    err = test(X, Y)
    test_cost = np.append(test_cost, err)
    if err < min_cost:
        W_h = w_h.get_value()
        W_o = w_o.get_value()
        B_h = b_h.get_value()
        B_o = b_o.get_value()
        min_cost = err

plt.figure()
plt.plot(range(noIters), test_cost)
plt.xlabel('iterations')
plt.ylabel('test error')
plt.title('Learning with %d hidden neurons'%opt_hidden)
plt.savefig('figure_t6.q1b_3.png')

w_h.set_value(W_h)
w_o.set_value(W_o)
b_h.set_value(B_h)
b_o.set_value(B_o)

pred = predict(X)

# plot trained and predicted points
fig = plt.figure()
ax = fig.gca(projection = '3d')
plot_original = ax.scatter(X[:,0], X[:,1], Y[:,0], marker='.', color = 'blue', label='test')
plot_pred = ax.scatter(X[:,0], X[:,1], pred[:,0], marker='.', color = 'red', label = 'predicted')
ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$y$')
ax.set_title('Test and Predicted Data Points')
ax.legend()
plt.savefig('figure_t6.q1b_4.png')
plt.show()
