# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

x_data = [1, 2, 3]
y_data = [1, 2, 3]

# variable : 변수 생성, random_normal : 랜덤 데이터 생성
# W = tf.Variable(tf.random.normal([1]), name = 'weight') #v2.0
W = tf.Variable(tf.random_normal([1]), name = 'weight')  #v1.0


# tensorflow 는 그래프를 미리 만들어 놓고 필요한 시점에 해당 그래프를 실행하는 지연실행(lazy evaluation) 방식을 사용
# placeholder : 변수의 타입을 미리 설정해놓고 필요한 변수를 나중에 받아서 실행. => tensorflow 2.0 은 session 이나 placeholder 사용하지 않음
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# hypothesis for linear model
hypothesis = X * W

# reduce_sum : 특정 차원을 제거하고 합계를 구한다
# reduce_mean : (특정 차원 제거) 행 단위로 평균을 냄
# square : 제곱
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # (hypothesis - y) 의 제곱에 평균을 냄 => cost


# Minimize: Gradient Descent using derivative: W -= learning_rate * derivative
learning_rate = 0.1
gradient = tf.reduce_mean((W * X - Y) * X)
descent = W - learning_rate * gradient
update = W.assign(descent)

# 그래프 그리기, 세션 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# update를 실행
for step in range(20):
    sess.run(update, feed_dict={X: x_data, Y: y_data})
    # cost 값과 Weight 값을 출
    print(step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W))

## step 이 증가할수록 cost 가 점점 작아짐, 그리고 W 값이 1에 가까워짐


# Minimize: Gradient Descent 간편히 쓸수 있는 함수, cost를 미분하지않아도됨.
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)





