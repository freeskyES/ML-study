# 대용량의 데이터를 읽어올땐 tensorflow 가 지원해주는 reader 를 쓴다. (numpy 로 불어오기 힘듬)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.set_random_seed(777)

# 1. filename들을 리스트화한다.
# 여러 파일도 넣을 수 있음. ['data-test.csv', 'data-test-02.csv']
filename_queue = tf.train.string_input_producer(
    ['../data/data-test.csv'], shuffle=False, name='filename_queue')

# 2. file 을 읽어올 reader를 정의해준다
# key 와 value 로 나누어서 읽겟다고 정의
reader = tf.TextLineReader()
key, value = reader.read(filename_queue)

# 3. value 값을 어떻게 파싱할 것인지 decode_csv 가 정의함
# 0. : float 값으로 정의
record_defaults = [[0.], [0.], [0.], [0.]]
xy = tf.decode_csv(value, record_defaults=record_defaults)

# batch
# 데이터를 읽어오게 함, 한번에 계산할 데이터
# tf.train.batch : 배치로 묶어서 리턴하는 함수
# batch로 묶고자 하는 tensor 들을 인자로 준다음에, 한번에 묶어서 리턴하고자 하는 텐서들의 갯수 batch_size 를 설정 (몇번 펌프해서 가져올지)
# xy[0:-1] : x, xy[-1:] : y   >> x y 데이터가 무엇인지 각각 설정해줌
train_x_batch, train_y_batch = \
    tf.train.batch([xy[0:-1], xy[-1:]], batch_size=10)

# step ? 학습 알고리즘의 학습 루프가 모델의 매개 변수를 업데이트 하기 위해 실행되는 횟수
# batch size ? 학습 알고리즘의 각 루프에서 공급하는 데이터 청크의 크기
# Epoch ? 학습 알고리즘에 피드를 제공하기 위해 데이터 세트 추출 배치를 실행하는 횟수


X = tf.placeholder(tf.float32, shape=[None, 3])  # 3: x shape 맞춰주기
Y = tf.placeholder(tf.float32, shape=[None, 1])  # 1: y shape

W = tf.Variable(tf.random_normal([3, 1]), name='weight')  # x, y
b = tf.Variable(tf.random_normal([1]), name='bias')       # y

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

# 큐 관리, session 넣어줌
# Start populating the filename queue.
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for step in range(2001):
    x_batch, y_batch = sess.run([train_x_batch, train_y_batch]) # 펌프해서 데이터를 가져옴
    cost_val, hy_val, _ = sess.run(
        [cost, hypothesis, train], feed_dict={X: x_batch, Y: y_batch})  # 학습할때 x, y batch 값을 가지고옴
    if step % 10 == 0:
        print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)

coord.request_stop()
coord.join(threads)

print("Your score will be ",
      sess.run(hypothesis, feed_dict={X: [[100, 70, 101]]}))

print("Other scores will be ",
      sess.run(hypothesis, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))

