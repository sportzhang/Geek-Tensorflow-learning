# -*- coding: utf-8 -*-
# @Time    : 2019/10/8 11:36
# @Author  : Fighter.kevin
# @E-mail  : gentle_kevin@163.com
# @File  : training_model.py

import pandas as pd
import numpy as np
import tensorflow as tf


def normalize_feature(df):
    return df.apply(lambda column: (column - column.mean()) / column.std())


df = normalize_feature(pd.read_csv('data1.csv', names=['square', 'bedrooms', 'price']))

ones = pd.DataFrame({'ones': np.ones(len(df))})  # ones是n行1列的数据框，表示x0恒为1
df = pd.concat([ones, df], axis=1)  # 根据列合并数据

# 数据处理：获取 X 和 y
X_data = np.array(df[df.columns[0:3]])
y_data = np.array(df[df.columns[-1]]).reshape(len(df), 1)

print(X_data.shape, type(X_data))
print(y_data.shape, type(y_data))

# 创建线性回归模型（数据流图）

alpha = 0.01  # 学习率 alpha
epoch = 500  # 训练全量数据集的轮数

# visualize loss
with tf.name_scope('input'):
    X = tf.placeholder(tf.float32, X_data.shape, name='X')
    y = tf.placeholder(tf.float32, y_data.shape, name='y')

with tf.name_scope('hypothesis'):
    # 权重变量 W，形状[3,1]
    W = tf.get_variable("weights",
                        (X_data.shape[1], 1),
                        initializer=tf.constant_initializer())
    # 假设函数 h(x) = w0*x0+w1*x1+w2*x2, 其中x0恒为1
    # 推理值 y_pred  形状[47,1]
    y_pred = tf.matmul(X, W, name='y_pred')

with tf.name_scope('loss'):
    loss_op = 1 / (2 * len(X_data)) * tf.matmul((y_pred - y), (y - y_pred), transpose_a=True)

with tf.name_scope('train'):
    train_op = tf.train.GradientDescentOptimizer(learning_rate=alpha).minimize(loss_op)


# 创建会话（运行环境）

with tf.Session() as sess:
    # 初始化全局变量
    sess.run(tf.global_variables_initializer())
    # 创建FileWriter实例，并传入当前会话加载的数据流图
    writer = tf.summary.FileWriter('./summary/linear-regression-2', sess.graph)
    # 记录所有损失值
    loss_data = []
    # 开始训练模型
    # 由于训练数据较小，所以采用批梯度优化算法，每次都是用全部数据进行训练
    for e in range(1, epoch + 1):
        _, loss, w = sess.run([train_op, loss_op, W], feed_dict={X:X_data, y:y_data})
        # 记录每一轮损失值变化情况
        loss_data.append(float(loss))
        if e % 100 == 0:
            log_str = "Epoch %d \t Loss=%.4g \t Model: y = %.4gx1 + %.4gx2 + %.4g"
            print(log_str % (e, loss, w[1], w[2], w[0]))

writer.close()

# 可视化损失值

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context="notebook", style="whitegrid", palette="dark")

ax = sns.lineplot(x='epoch', y='loss', data=pd.DataFrame({'loss': loss_data, 'epoch': np.arange(epoch)}))
ax.set_xlabel('epoch')
ax.set_ylabel('loss')
plt.show()