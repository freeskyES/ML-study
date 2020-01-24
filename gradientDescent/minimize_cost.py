# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.Variable(-3.0)  #v1.0


# hypothesis for linear model
hypothesis = X * W

# reduce_sum : 특정 차원을 제거하고 합계를 구한다
# reduce_mean : (특정 차원 제거) 행 단위로 평균을 냄
# square : 제곱
cost = tf.reduce_mean(tf.square(hypothesis - Y))  # (hypothesis - y) 의 제곱에 평균을 냄 => cost

# Minimize: Gradient Descent 간편히 쓸수 있는 함수, cost를 미분하지않아도됨.
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
train = optimizer.minimize(cost)


# 그래프 그리기, 세션 초기화
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# update를 실행
for step in range(100):
    print(step, sess.run(W))
    sess.run(train)

## step 이 증가할수록 cost 가 점점 작아짐, 그리고 W 값이 1에 가까워짐







