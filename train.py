import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import utils as tu
from config import cfg
from LeeNet import LeeNet

x_train, y_train = tu.load_data(True)

# GPU 메모리 증가 허용
config = tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))


# initialize
sess = tf.Session(config=config)
model = LeeNet(sess, "LeeNet")
writer = tf.summary.FileWriter(cfg.log_dir + '/LeeNet')
writer.add_graph(sess.graph)
sess.run(tf.global_variables_initializer())


# train my model
step = 0
costs = []
train_accu = []

print('Learning Started!')

for epoch in range(cfg.epoch):
    avg_cost = 0
    avg_accu = 0
    total_batch = int(x_train.shape[0] / cfg.b_size)
    minibatches = tu.random_mini_batches(x_train, y_train, cfg.b_size)

    for i in minibatches:
        (batch_xs, batch_ys) = i
        _, temp_cost, temp_accu, summary = model.train(batch_xs, batch_ys)
        avg_cost += temp_cost / total_batch
        avg_accu += temp_accu / total_batch
        writer.add_summary(summary, global_step=step)
        step += 1

    costs.append(avg_cost)
    train_accu.append(avg_accu)

    print('Epoch', '%04d' % (epoch + 1),
          ': cost =', '{:.9f}'.format(avg_cost), '| accuracy =', '{:.9f}'.format(avg_accu))

print('Learning Finished!')


# Show plots of costs and train accuracy
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('epochs')
plt.title("LeeNet Model Costs")
plt.show()

plt.plot(np.squeeze(train_accu))
plt.ylabel('train accuracy')
plt.xlabel('epochs')
plt.title("LeeNet Model Train accuracy")
plt.show()


# Save model
saver = tf.train.Saver()
saver.save(sess, cfg.save)
print("Model saved in file: ", cfg.save)
