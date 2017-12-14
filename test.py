import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

import utils as tu
from config import cfg
from LeeNet import LeeNet

x_test, y_test = tu.load_data(False)

CLASSES = {}
labeldict = tu.unpickle('./input/batches.meta')
for i in range(10):
    CLASSES[i] = str(labeldict.get(b'label_names')[i])[2:-1]

# GPU 메모리 증가 허용
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))

# initialize
sess = tf.Session(config=config)
model = LeeNet(sess, "LeeNet")

# load save model checkpoint
saver = tf.train.Saver()
saver.restore(sess, cfg.save)

# Test model and check accuracy
print('Test Accuracy:', model.evaluate(x_test, y_test))

# Get one and predict
r = random.randint(0, x_test.shape[0] - 1)

actualLabel = (sess.run(tf.argmax(y_test[r:r + 1], 1)))[0]
print("\nLabel: ", actualLabel, '-', CLASSES[actualLabel])
predictedLabel = (sess.run(tf.argmax(model.predict(x_test[r:r + 1]), 1)))[0]
print("Prediction: ", predictedLabel, '-', CLASSES[predictedLabel])
plt.imshow(x_test[r,:,:,:])
plt.show()
