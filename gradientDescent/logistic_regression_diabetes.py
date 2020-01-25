import tensorflow.compat.v1 as tf
import numpy as np
tf.disable_v2_behavior()
tf.set_random_seed(777)

xy = np.loadtxt('../data/data_diabetes.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1] # 마지막 결과 열을 뺀 나머지 모두
y_data = xy[:, [-1]]

print(x_data.shape, y_data.shape)

X = tf.placeholder(tf.float32, shape=[None, 8])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([8, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# tf.sigmoid = tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
# dtype 을 float32 로 설정햇으므로, True = 1.0, False = 0.0 으로 캐스팅되 결과가 반환됨
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
# 예측한 값들이 얼마나 정확한지 계산해보기 (0 또는 1값 나온 리스트를 평균을 냄)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Launch graph
with tf.Session() as sess:
    # Initialize TensorFlow variables
    sess.run(tf.global_variables_initializer())

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})
        # 200번 마다 출력
        if step % 200 == 0:
            print(step, cost_val)

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                       feed_dict={X: x_data, Y: y_data})
    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)


'''
cost_val 값이 점점 내려감

Accuracy:  0.7628459
76% 정확도 나옴
'''
