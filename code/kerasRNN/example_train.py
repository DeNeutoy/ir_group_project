#imports

import numpy as np
import os

import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation,Reshape
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.optimizers import Adam


if __name__ == "__main__":
    os.chdir("../../")


# set up model
nFeatures = 1
nHidden = 2
input_len = 10
nSim = 100
nT_in = 10
nT_out = 10
nIterations = 10000




model = Sequential()
model.add(SimpleRNN(nHidden, input_shape=(nT_in,nFeatures), activation='tanh', return_sequences=False))
# model.add(LSTM(nT_out*nFeatures+ nHidden, input_shape=(nT_in,nFeatures), activation='tanh', return_sequences=False))
# model.add(Dropout(.2))
model.add(Dense(nT_out*nFeatures))
model.add(Dense(nT_out*nFeatures))
model.add(Dense(nT_out*nFeatures))
model.add(Dense(nT_out*nFeatures))
model.add(Reshape((nT_out,nFeatures)))
# model.add(Activation('linear'))
# model.add(SimpleRNN(16))

adam = Adam(lr=0.01)
model.compile(optimizer=adam, loss='mse')


np.random.seed(12345)

nT = nT_in + nT_out
paths = np.zeros((nSim,nT,nFeatures))
for iSim in range(nSim):
    xPrev = np.random.rand()
    alpha = 0.6*np.random.rand() + 0.7
    for t in range(nT):
        x = alpha*xPrev +  np.sin(2*np.pi*float(t)/float(nT_in)) + 0.2*np.random.randn()
        paths[iSim,t,0] = x
        xPrev = x

plt.plot(range(nT),paths[0,:,0],"-o")
plt.plot(range(nT),paths[1,:,0],"-o")
plt.show()


X_train = paths[:nSim/2,:nT_in,:]
Y_train = paths[:nSim/2,nT_in:,:]
X_test = paths[nSim/2:,:nT_in,:]
Y_test = paths[nSim/2:,nT_in:,:]




# if False:
#     for chan in range(3):
#         for epoch in range(1,trials):
#             X_train[epoch/2+5, epoch/trials, chan] = 10
#             X_test[epoch/2+5, epoch/trials, chan] = 10
#
# _ = plt.imshow(np.mean(X_test, 2))



print('-'*50)
print('Training model...')
print('-'*50)

for iIteration in range(nIterations):
    print('-'*50)
    print('Iteration {0}/{1}'.format(iIteration + 1, nIterations))
    print('-'*50)

    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),
              batch_size=50, nb_epoch=1)
    pred = model.predict(X_test)

print X_test[0,:,0]
print Y_test[0,:,0]
print pred[0,:,0]
print X_test[1,:,0]
print Y_test[1,:,0]
print pred[1,:,0]

plt.plot(range(nT),np.concatenate((X_test[0,:,0],Y_test[0,:,0]),axis=1),'b-')
plt.plot(range(nT_in,nT),pred[0,:,0],'bo')
plt.plot(range(nT),np.concatenate((X_test[1,:,0],Y_test[1,:,0]),axis=1),'r-')
plt.plot(range(nT_in,nT),pred[1,:,0],'ro')
plt.show()

print "!!!!!"

print "End."