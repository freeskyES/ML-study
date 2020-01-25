import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(777)

x_data = [[1, 2],
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0],
          [0],
          [0],
          [1],
          [1],
          [1]]

X = tf.placeholder(tf.float32, shape=[None, 2])
Y = tf.placeholder(tf.float32, shape=[None, 1])

# Weight : 들어오는 값 x, 나가는 값 y 의 갯수 [2,1]
W = tf.Variable(tf.random_normal([2, 1]), name='weight')
# bias : 나가는 값의 갯수와 같음 y, 1개
b = tf.Variable(tf.random_normal([1]), name='bias')

# tf.sigmoid = tf.div(1., 1. + tf.exp(tf.matmul(X, W) + b))
hypothesis = tf.sigmoid(tf.matmul(X, W) + b)

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y)* tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

# Accuracy computation
# True if hypothesis>0.5 else False
# dtype 을 float32 로 설정햇으므로, True = 1.0, False = 0.0 으로 캐스팅되 결과가 반환됨
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
# 예측한 값들이 얼마나 정확한지 계산해보기 (0 또는 1값 나온 리스트를 평균을 냄)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype = tf.float32))


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
Hypothesis:  [[0.03074028] >>0
 [0.15884678]  >>0
 [0.30486736]  >>0
 [0.78138196]  >>1
 [0.93957496]  >>1
 [0.9801688 ]] >>1
 
 Correct (Y):  [[ 0.]
 [ 0.]
 [ 0.]
 [ 1.]
 [ 1.]
 [ 1.]]
 
 Accuracy:  1.0
 결과로, predicted 값과 실제 Y data 와 일치 >> 잘나옴
 
 
'''

