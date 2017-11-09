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
d = T.fmatrix('d')


rng = np.random.RandomState(123)
theano_rng = RandomStreams(rng.randint(2 ** 30))

corruption_level=0.1
training_epochs = 2
learning_rate = 0.1
batch_size = 128

#first layer
W1 = init_weights(28*28, 900)
b1 = init_bias(900)

b1_prime = init_bias(900)
W1_prime = W1.transpose()
#second layer
W2 = init_weights(900, 625)
b2 = init_bias(625)

b2_prime = init_bias(625)
W2_prime = W2.transpose()
#third layer
W3 = init_weights(625, 400)
b3 = init_bias(400)

b3_prime = init_bias(625)
W3_prime = W3.transpose()
#output
W_end = init_weights(900, 10)
b_end = init_bias(10)

print("lmao familicious")

tilde_x = theano_rng.binomial(size=x.shape, n=1, p=1 - corruption_level,
                              dtype=theano.config.floatX)*x
y1 = T.nnet.sigmoid(T.dot(tilde_x, W1) + b1)
z1 = T.nnet.sigmoid(T.dot(y1, W1_prime) + b1_prime)
cost1 = - T.mean(T.sum(x * T.log(z1) + (1 - x) * T.log(1 - z1), axis=1))

y2 = T.nnet.sigmoid(T.dot(y1, W2) + b2)
z2 = T.nnet.sigmoid(T.dot(y2, W2_prime) + b2_prime)
cost2 = - T.mean(T.sum(y1 * T.log(z2) + (1 - y1) * T.log(1 - z2), axis=1))

y3 = T.nnet.sigmoid(T.dot(y2, W3) + b3)
z3 = T.nnet.sigmoid(T.dot(y3, W3_prime) + b3_prime)
cost3 = - T.mean(T.sum(y2 * T.log(z3) + (1 - y2) * T.log(1 - z3), axis=1))

#first layer
params1 = [W1, b1, b1_prime]
grads1 = T.grad(cost1, params1)
updates1 = [(param1, param1 - learning_rate * grad1)
           for param1, grad1 in zip(params1, grads1)]
train_da1 = theano.function(inputs=[x], outputs = cost1, updates = updates1, allow_input_downcast = True)
#second layer
params2 = [W2, b2,b2_prime]
grads2 = T.grad(cost2, params2)
updates2 = [(param2, param2 - learning_rate * grad2)
           for param2, grad2 in zip(params2, grads2)]
train_da2 = theano.function(inputs=[y1], outputs = cost2, updates = updates2, allow_input_downcast = True)
#third layer
params3 = [W3, b3,b3_prime]
grads3 = T.grad(cost3, params3)
updates3 = [(param3, param3 - learning_rate * grad3)
           for param3, grad3 in zip(params3, grads3)]
train_da3 = theano.function(inputs=[y2], outputs = cost3, updates = updates3, allow_input_downcast = True)



# p_y2_ffn = T.nnet.softmax(T.dot(y1, W_end)+b_end)
# y2_ffn = T.argmax(p_y2_ffn, axis=1)
# cost2_ffn = T.mean(T.nnet.categorical_crossentropy(p_y2_ffn, d))
# params2_ffn = [W1, b1, b1_prime]
# grads2_ffn = T.grad(cost2_ffn, params2_ffn)
# updates2_ffn = [(param2_ffn, param2_ffn - learning_rate * grad2_ffn)
#            for param2_ffn, grad2_ffn in zip(params2_ffn, grads2_ffn)]
# train_ffn = theano.function(inputs=[x, d1], outputs = cost2_ffn, updates = updates2_ffn, allow_input_downcast = True)
# test_ffn = theano.function(inputs=[x], outputs = y2_ffn, allow_input_downcast=True)


print('training dae1 ...')
d1 = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da1(trX[start:end]))
    d1.append(np.mean(c, dtype='float64'))
    print(d[epoch])

print('training dae2 ...')
d2 = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da2(trX[start:end]))
    d2.append(np.mean(c, dtype='float64'))
    print(d[epoch])

print('training dae3 ...')
d3 = []
for epoch in range(training_epochs):
    # go through trainng set
    c = []
    for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
        c.append(train_da3(trX[start:end]))
    d3.append(np.mean(c, dtype='float64'))
    print(d[epoch])



pylab.figure()
pylab.plot(range(training_epochs), d3)
pylab.xlabel('iterations')
pylab.ylabel('cross-entropy')
pylab.savefig('figure_2b_1.png')

w1 = W1.get_value()
pylab.figure()
pylab.gray()
for i in range(100):
    pylab.subplot(10, 10, i+1); pylab.axis('off'); pylab.imshow(w1[:,i].reshape(28,28))
pylab.savefig('figure_2b_2.png')


# print('\ntraining ffn ...')
# d_ffn, a = [], []
# for epoch in range(training_epochs):
#     # go through trainng set
#     c = []
#     for start, end in zip(range(0, len(trX), batch_size), range(batch_size, len(trX), batch_size)):
#         c.append(train_ffn(trX[start:end], trY[start:end]))
#     d_ffn.append(np.mean(c, dtype='float64'))
#     a.append(np.mean(np.argmax(teY, axis=1) == test_ffn(teX)))
#     print(a[epoch])
#
# pylab.figure()
# pylab.plot(range(training_epochs), d_ffn)
# pylab.xlabel('iterations')
# pylab.ylabel('cross-entropy')
# pylab.savefig('figure_2b_3.png')
#
# pylab.figure()
# pylab.plot(range(training_epochs), a)
# pylab.xlabel('iterations')
# pylab.ylabel('test accuracy')
# pylab.savefig('figure_2b_4.png')
# pylab.show()
#
# w_end = W_end.get_value()
# pylab.figure()
# pylab.gray()
# pylab.axis('off'); pylab.imshow(w_end)
# pylab.savefig('figure_2b_5.png')
