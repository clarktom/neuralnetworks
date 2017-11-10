from load import mnist
import numpy as np

import pylab

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# 1 encoder, decoder and a softmax layer

def init_weights(n_visible, n_hidden):
    initial_W = np.asarray(
        np.random.uniform(
            low=-4 * np.sqrt(6. / (n_hidden + n_visible)),
            high=4 * np.sqrt(6. / (n_hidden + n_visible)),
            size=(n_visible, n_hidden)),
        dtype=theano.config.floatX)
    return theano.shared(value=initial_W, name='W', borrow=True)

def init_bias(n):
    return theano.shared(value=np.zeros(n,dtype=theano.config.floatX),borrow=True)

trX, teX, trY, teY = mnist()

trX, trY = trX[:12000], trY[:12000]
teX, teY = teX[:2000], teY[:2000]

x = T.fmatrix('x')
d1 = T.fmatrix('d1')

rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

corruption_level=0.1
training_epochs = 2
learning_rate = 0.1
batch_size = 128

#first layer
W1 = init_weights(28*28, 900)
b1 = init_bias(900)

b1_prime = init_bias(28*28)
W1_prime = W1.transpose()
#second layer
W2 = init_weights(900, 625)
b2 = init_bias(625)

b2_prime = init_bias(900)
W2_prime = W2.transpose()
#third layer
W3 = init_weights(625, 400)
b3 = init_bias(400)

b3_prime = init_bias(625)
W3_prime = W3.transpose()
#fourth layer
W4 = init_weights(400, 100)
b4 = init_bias(100)

b4_prime = init_bias(400)
W4_prime = W4.transpose()
#output
W_ffn = init_weights(100, 10)
b_ffn = init_bias(10)

print("lmao familicious fam")

tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x
y1 = T.nnet.sigmoid(T.dot(x, W1) + b1)
z1_1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
cost1 = - T.mean(T.sum(x * T.log(z1_1) + (1 - x) * T.log(1 - z1_1), axis=1))

yy1 = T.fmatrix('yy1')
tilde_y1 = theano_rng.binomial(size=yy1.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*yy1
y2 = T.nnet.sigmoid(T.dot(tilde_y1, W2) + b2)
z2_2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)
z1_2 = T.nnet.sigmoid(T.dot(z2_2, W1_prime) + b1_prime)
cost2 = - T.mean(T.sum(x * T.log(z1_2) + (1 - x) * T.log(1 - z1_2), axis=1))

yy2 = T.fmatrix('yy2')
tilde_y2 = theano_rng.binomial(size=yy2.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*yy2
y3 = T.nnet.sigmoid(T.dot(tilde_y2, W3) + b3)
z3_3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)
z2_3 = T.nnet.sigmoid(T.dot(z3_3, W2_prime) + b2_prime)
z1_3 = T.nnet.sigmoid(T.dot(z2_3, W1_prime) + b1_prime)
cost3 = - T.mean(T.sum(x * T.log(z1_3) + (1 - x) * T.log(1 - z1_3), axis=1))

yy3 = T.fmatrix('yy3')
tilde_y3 = theano_rng.binomial(size=yy3.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*yy3
y4 = T.nnet.sigmoid(T.dot(tilde_y3, W4) + b4)
z4_4 = T.nnet.sigmoid(T.dot(y4, W4_prime) + b4_prime)
z3_4 = T.nnet.sigmoid(T.dot(z4_4, W3_prime) + b3_prime)
z2_4 = T.nnet.sigmoid(T.dot(z3_4, W2_prime) + b2_prime)
z1_4 = T.nnet.sigmoid(T.dot(z2_4, W1_prime) + b1_prime)
cost4 = - T.mean(T.sum(x * T.log(z1_4) + (1 - x) * T.log(1 - z1_4), axis=1))

#first layer
params1 = [W1, b1, b1_prime]
grads1 = T.grad(cost1, params1)
updates1 = [(param1, param1 - learning_rate * grad1)
           for param1, grad1 in zip(params1, grads1)]
train_da1 = theano.function(inputs=[x], outputs=cost1, updates=updates1, allow_input_downcast=True)
test_da1 = theano.function(inputs=[x], outputs=[y1, z1_1], updates=None, allow_input_downcast=True)

#second layer
params2 = [W2, b2, b2_prime]
grads2 = T.grad(cost2, params2)
updates2 = [(param2, param2 - learning_rate * grad2)
           for param2, grad2 in zip(params2, grads2)]
train_da2 = theano.function(inputs=[x, yy1], outputs=cost2, updates=updates2, allow_input_downcast=True)
test_da2 = theano.function(inputs=[yy1], outputs=[y2, z1_2], updates=None, allow_input_downcast=True)

#third layer
params3 = [W3, b3, b3_prime]
grads3 = T.grad(cost3, params3)
updates3 = [(param3, param3 - learning_rate * grad3)
           for param3, grad3 in zip(params3, grads3)]
train_da3 = theano.function(inputs=[x, yy2], outputs = cost3, updates = updates3, allow_input_downcast = True)
test_da3 = theano.function(inputs=[yy2], outputs = [y3, z1_3], updates = None, allow_input_downcast = True)

#fourth layer
params4 = [W4, b4, b4_prime]
grads4 = T.grad(cost4, params4)
updates4 = [(param4, param4 - learning_rate * grad4)
           for param4, grad4 in zip(params4, grads4)]
train_da4 = theano.function(inputs=[x, yy3], outputs = cost4, updates = updates4, allow_input_downcast = True)
test_da4 = theano.function(inputs=[yy3], outputs = [y4, z1_4], updates = None, allow_input_downcast = True)

#softmax layer
yy4 = T.fmatrix('yy4')
p_y_ffn = T.nnet.softmax(T.dot(yy4, W_ffn)+b_ffn)
y_ffn = T.argmax(p_y_ffn, axis=1)
cost_ffn = T.mean(T.nnet.categorical_crossentropy(p_y_ffn, d1))
params_ffn = [W_ffn, b_ffn]
grads_ffn = T.grad(cost_ffn, params_ffn)
updates_ffn = [(param_ffn, param_ffn - learning_rate * grad_ffn)
            for param_ffn, grad_ffn in zip(params_ffn, grads_ffn)]
train_ffn = theano.function(inputs=[x, d1], outputs = cost_ffn, updates = updates_ffn, allow_input_downcast = True)
test_ffn = theano.function(inputs=[x], outputs = y_ffn, allow_input_downcast=True)

print('training dae1 ...')
d1 = []
for epoch in range(training_epochs):

    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        cost = train_da1(trX[start:end])
        c.append(cost)
    d1.append(np.mean(c, dtype='float64'))
    print(d[epoch])

print('training dae2 ...')
d2 = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        yy1, _ = test_da1(trX[start:end])
        cost = train_da2(trX[start:end], yy1)
        c.append(cost)
    d2.append(np.mean(c, dtype='float64'))
    print(d[epoch])

print('training dae3 ...')
d3 = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        yy1, _ = test_da1(trX[start:end])
        yy2, _ = test_da2(yy1)
        cost = train_da3(trX[start:end], yy2)
        c.append(cost)
    d3.append(np.mean(c, dtype='float64'))
    print(d[epoch])

print('training dae4 ...')
d4 = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        yy1, _ = test_da1(trX[start:end])
        yy2, _ = test_da2(yy1)
        yy3, _ = test_da2(yy2)
        cost = train_da4(trX[start:end], yy3)
        c.append(cost)
    d4.append(np.mean(c, dtype='float64'))
    print(d[epoch])
#softmax
print('\ntraining ffn ...')
testAccuracy = []
trainCost = []
for epoch in range(training_epochs):
        # go through trainng set
        for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
            cost = train_ffn(trX[start:end], trY[start:end])
        testAccuracy.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
        trainCost.append(cost/(len(trX) // batch_size))
        if epoch%100 == 0:
        	print(testAccuracy[epoch])
#learning curves
pylab.figure()
pylab.plot(range(training_epochs), d1)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.title('first layer')
pylab.savefig('firstLayer')

pylab.figure()
pylab.plot(range(training_epochs), d2)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.title('second layer')
pylab.savefig('secondLayer')

pylab.figure()
pylab.plot(range(training_epochs), d3)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.title('third layer')
pylab.savefig('thirdLayer')

pylab.figure()
pylab.plot(range(training_epochs), testAccuracy)
pylab.xlabel('iterations')
pylab.ylabel('test accuracy')
pylab.title('test accuracy')
pylab.savefig('testAcc')

pylab.figure()
pylab.plot(range(training_epochs), trainCost)
pylab.xlabel('iterations')
pylab.ylabel('training cost')
pylab.title('training cost')
pylab.savefig('trainC')

#weights
w1 = W1.get_value()
pylab.figure('first layer weights')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w1[:,i].reshape(28,28))

pylab.savefig('firstLayerWeights')

w2 = W2.get_value()
pylab.figure('second layer weights')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w2[:,i].reshape(30,30))
pylab.savefig('secondLayerWeights')

w3 = W3.get_value()
pylab.figure('third layer weights')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w3[:,i].reshape(25,25))

pylab.savefig('thirdLayerWeights')

#reconstructed images
pylab.figure('original images')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(teX[i,:].reshape(28,28))

pylab.savefig('Inp_image_original')
yy1, zz1 = test_da1(teX[:100])
yy2, zz2 = test_da2(yy1)
yy3, zz3 = test_da3(yy2)

pylab.figure('reconstructed image first layer')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(zz1[i,:].reshape(28,28))
pylab.savefig('Inp_image_firstLayer')

pylab.figure('reconstructed image second layer')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(zz2[i,:].reshape(28,28))
pylab.savefig('Inp_image_secondLayer')

pylab.figure('reconstructed image third layer')
pylab.gray()
for i in range(100):

    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(zz3[i,:].reshape(28,28))

pylab.savefig('Inp_image_thirdLayer')

pylab.figure('first hidden activation')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(yy1[i,:].reshape(30,30))
pylab.savefig('activation_firstLayer')

pylab.figure('second hidden activation')
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(yy2[i,:].reshape(25,25))
pylab.savefig('activation_secondLayer')

pylab.figure('third hidden activation')
pylab.gray()
for i in range(100):

    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(yy3[i,:].reshape(20,20))
pylab.savefig('activation_thirdLayer')
pylab.show()
