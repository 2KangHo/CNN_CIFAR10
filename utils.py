import os
import numpy as np
import tensorflow as tf

from config import cfg

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_data(is_training=True):
    if is_training:
        data1 = unpickle(cfg.data1)
        data2 = unpickle(cfg.data2)
        data3 = unpickle(cfg.data3)
        data4 = unpickle(cfg.data4)
        data5 = unpickle(cfg.data5)

        datas = np.concatenate((data1.get(b'data'),
                                data2.get(b'data'),
                                data3.get(b'data'),
                                data4.get(b'data'),
                                data5.get(b'data')), axis=0)
        labels = np.concatenate((data1.get(b'labels'),
                                 data2.get(b'labels'),
                                 data3.get(b'labels'),
                                 data4.get(b'labels'),
                                 data5.get(b'labels')), axis=0)

        trX = np.empty((50000, 3072))
        trX[:,:3072:3] = datas[:,:1024]
        trX[:,1:3072:3] = datas[:,1024:2048]
        trX[:,2:3072:3] = datas[:,2048:]

        trX = trX.reshape(-1, 32, 32, 3).astype('float32') / 255.
        trY = labels.reshape(-1, 1)

        onehotY = np.zeros((50000, 10))
        for i in range(50000):
            onehotY[i,trY[i,0]] = 1
        trY = onehotY

        return trX, trY
    else:
        testdata = unpickle(cfg.testdata)

        x_testdata = testdata.get(b'data')
        y_testdata = testdata.get(b'labels')

        teX = np.empty((10000, 3072))
        teX[:,:3072:3] = x_testdata[:,:1024]
        teX[:,1:3072:3] = x_testdata[:,1024:2048]
        teX[:,2:3072:3] = x_testdata[:,2048:]

        teX = teX.reshape(-1, 32, 32, 3).astype('float32') / 255.
        teY = np.zeros((10000, 1))
        teY[:,0] = y_testdata[:]

        onehot = np.zeros((10000, 10))
        for i in range(10000):
            onehot[i,int(teY[i,0])] = 1
        teY = onehot

        return teX, teY

def random_mini_batches(X_train, Y_train, minibatch_size):
    data_size = Y_train.shape[0]
    
    minibatches = []
    
    num = int(data_size / minibatch_size)
    num_ex = data_size % minibatch_size
    
    if (num_ex > 0):
        num = num + 1
    
    for i in range(num):
        inds = np.random.randint(0,int(data_size),size=int(minibatch_size))
        x_batch, y_batch = X_train[inds,...], Y_train[inds,...]
        minibatches.append((x_batch,y_batch))
        
    return minibatches