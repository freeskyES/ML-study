import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
tf.set_random_seed(777)  # for reproducibility

xy = np.loadtxt('../data/data-test.csv', delimiter=',', dtype=np.float32)
# : 전체
# 0: -1  >> 처음부터 ~마지막행을 제외한것까지
x_data = xy[:, 0:-1] # n 행 전부다, 마지막제외한것의 열 모두
y_data = xy[:, [-1]] # n 행, 마지막열만 가져오기

print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)


X = tf.placeholder(tf.float32, shape=[None, 3])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([3, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost/loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# Minimize
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

# Launch the graph in a session.
sess = tf.Session()
# Initializes global variables in the graph.
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, hy_val, _ = sess.run([cost, hypothesis, train],
                                   feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)


# score 를 예측한 값
print("Your score will be ", sess.run(hypothesis,
                                      feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ", sess.run(hypothesis,
                                        feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))