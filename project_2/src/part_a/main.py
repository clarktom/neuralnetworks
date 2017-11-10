from load import mnist
import numpy as np
import pylab

import theano
from theano import tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import pool

# 1 convolution layer, 1 max pooling layer and a softmax layer

np.random.seed(10)
batch_size = 128
noIters = 25
learningrate = 0.05
decayparameter = 1e-4
momentum = 0.1
decayparameterRMS = 1e-4
p = 0.9
ebs = 1e-6
learningrateRMS = 0.001


def init_weights_bias4(filter_shape, d_type):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def init_weights_bias2(filter_shape, d_type):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    return theano.shared(w_values,borrow=True), theano.shared(b_values, borrow=True)

def set_weights_bias4(filter_shape, d_type, w, b):
    fan_in = np.prod(filter_shape[1:])
    fan_out = filter_shape[0] * np.prod(filter_shape[2:])

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[0],), dtype=d_type)
    w.set_value(w_values), b.set_value(b_values)
    return

def set_weights_bias2(filter_shape, d_type, w, b):
    fan_in = filter_shape[1]
    fan_out = filter_shape[0]

    bound = np.sqrt(6. / (fan_in + fan_out))
    w_values =  np.asarray(
            np.random.uniform(low=-bound, high=bound, size=filter_shape),
            dtype=d_type)
    b_values = np.zeros((filter_shape[1],), dtype=d_type)
    w.set_value(w_values), b.set_value(b_values)
    return

def model(X, w1, b1, w2, b2, w3, b3, w4, b4):
    y1 = T.nnet.relu(conv2d(X, w1) + b1.dimshuffle('x', 0, 'x', 'x'))
    pool_dim = (2, 2)
    o1 = pool.pool_2d(y1, pool_dim, mode='max')

    y2 = T.nnet.relu(conv2d(o1, w2) + b2.dimshuffle('x', 0, 'x', 'x'))
    o2 = pool.pool_2d(y2, pool_dim, mode='max')
    o3 = T.flatten(o2, outdim=2)

    y3 = T.nnet.sigmoid(T.dot(o3, w3) + b3)
    pyx = T.nnet.softmax(T.dot(y3, w4) + b4)

    return y1, o1, y2, o2, pyx

def sgd(cost, params, lr=0.05, decay=0.0001):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        updates.append([p, p - (g + decay*p) * lr])
    return updates

def sgd_momentum(cost, params, lr=0.05, decay=0.0001, momentum=0.5):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        v = theano.shared(p.get_value()*0.)
        v_new = momentum*v - (g + decay*p) * lr
        updates.append([p, p + v_new])
        updates.append([v, v_new])
    return updates

def RMSprop(cost, params, lr=0.001, decay=0.0001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * (g+ decay*p)))
    return updates


def shuffle_data (samples, labels):
    idx = np.arange(samples.shape[0])
    np.random.shuffle(idx)
    samples, labels = samples[idx], labels[idx]
    return samples, labels

trX, teX, trY, teY = mnist(onehot=True)

trX = trX.reshape(-1, 1, 28, 28)
teX = teX.reshape(-1, 1, 28, 28)

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]


X = T.tensor4('X')
Y = T.matrix('Y')
print('xd200')
num_filters1 = 15
num_filters2 = 20
w1, b1 = init_weights_bias4((num_filters1, 1, 9, 9), X.dtype)
w2, b2 = init_weights_bias4((num_filters2, num_filters1, 5, 5), X.dtype)
w3, b3 = init_weights_bias2((num_filters2*3*3, 100), X.dtype)
w4, b4 = init_weights_bias2((100, 10), X.dtype)

y1, o1, y2, o2, py_x  = model(X, w1, b1, w2, b2, w3, b3, w4, b4)

y_x = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
params = [w1, b1, w2, b2, w3, b3, w4, b4]

updates = sgd(cost, params, learningrate, decayparameter)
updates2 = sgd_momentum(cost, params, learningrate, decayparameter, momentum)
updates3 = RMSprop(cost, params, learningrateRMS, decayparameterRMS, p, ebs)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
train2 = theano.function(inputs=[X, Y], outputs=cost, updates=updates2, allow_input_downcast=True)
train3 = theano.function(inputs=[X, Y], outputs=cost, updates=updates3, allow_input_downcast=True)

predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
test = theano.function(inputs = [X], outputs=[y1, o1, y2, o2], allow_input_downcast=True)


a = []
trainCost = []
a2 = []
trainCost2 = []
a3 = []
trainCost3 = []

for i in range(noIters):
    trX, trY = shuffle_data (trX, trY)
    teX, teY = shuffle_data (teX, teY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train(trX[start:end], trY[start:end])
    a.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    trainCost.append(cost/(len(trX) // batch_size))
    print(a[i])



print('sgd with momentum ..')
set_weights_bias4((num_filters1, 1, 9, 9), X.dtype, w1, b1)
set_weights_bias4((num_filters2, num_filters1, 5, 5), X.dtype, w2, b2)
set_weights_bias2((num_filters2*3*3, 100), X.dtype, w3, b3)
set_weights_bias2((100, 10), X.dtype, w4, b4)

for i in range(noIters):
    trX, trY = shuffle_data (trX, trY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train2(trX[start:end], trY[start:end])
    a2.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    trainCost2.append(cost/(len(trX) // batch_size))
    print(a[i])


print('RMSprop ..')
set_weights_bias4((num_filters1, 1, 9, 9), X.dtype, w1, b1)
set_weights_bias4((num_filters2, num_filters1, 5, 5), X.dtype, w2, b2)
set_weights_bias2((num_filters2*3*3, 100), X.dtype, w3, b3)
set_weights_bias2((100, 10), X.dtype, w4, b4)

for i in range(noIters):
    trX, trY = shuffle_data (trX, trY)
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train3(trX[start:end], trY[start:end])
    a3.append(np.mean(np.argmax(teY, axis=1) == predict(teX)))
    trainCost3.append(cost/(len(trX) // batch_size))
    print(a[i])

pylab.figure()
pylab.plot(range(noIters), a, label='SGD')
pylab.plot(range(noIters), a2, label='SGD with momentum')
pylab.plot(range(noIters), a3, label='RMSprop')
pylab.xlabel('epochs')
pylab.ylabel('test accuracy')
pylab.legend(loc='lower right')
pylab.title('test accuracy ')
pylab.savefig('testAccuracy')

pylab.figure()
pylab.plot(range(noIters), trainCost, label='SGD')
pylab.plot(range(noIters), trainCost2, label='SGD with momentum')
pylab.plot(range(noIters), trainCost3, label='RMSprop')
pylab.xlabel('epochs')
pylab.ylabel('training cost')
pylab.legend(loc='upper right')
pylab.title('training cost')
pylab.savefig('trainingCost')

w = w1.get_value()
pylab.figure()
pylab.gray()
for i in range(num_filters1):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(w[i,:,:,:].reshape(9,9))
pylab.title('filters learned')
pylab.savefig('filtersLearned')

ind = np.random.randint(low=0, high=2000)
convolved, pooled, convolved2, pooled2 = test(teX[ind:ind+1,:])

pylab.figure()
pylab.gray()
pylab.axis('off'); pylab.imshow(teX[ind,:].reshape(28,28))
pylab.title('input image')
pylab.savefig('inputImage')

pylab.figure()
pylab.gray()
for i in range(num_filters1):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(convolved[0,i,:].reshape(20,20))
pylab.title('first convolved feature maps')
pylab.savefig('1stConv_layer')

pylab.figure()
pylab.gray()
for i in range(5):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(pooled[0,i,:].reshape(10,10))
pylab.title('first pooled feature maps')
pylab.savefig('1stPool_layer')

pylab.figure()
pylab.gray()
for i in range(num_filters2):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(convolved2[0,i,:].reshape(6,6))

pylab.title('second convolved feature maps')
pylab.savefig('2stConv_layer')
pylab.figure()
pylab.gray()
for i in range(5):
    pylab.subplot(5, 5, i+1); pylab.axis('off'); pylab.imshow(pooled2[0,i,:].reshape(3,3))
pylab.title('second pooled feature maps')
pylab.savefig('2stPool_layer_SGD')

pylab.show()
