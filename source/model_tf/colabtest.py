import tensorflow as tf

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [3, 4, 7, 8, 11, 12, 15, 16, 19, 20]

X = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])
W = tf.Variable(tf.random_normal([1, 1]))
b = tf.Variable(tf.random_normal([1]))

logit = W*X+b
loss = tf.reduce_mean(tf.square(Y-logit))
trainer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(60000):
  _, l = sess.run([trainer, loss], feed_dict={X:x, Y:y})
  if i % 500 == 0:
    print(i, l)

a = sess.run(logit, feed_dict={X:x})
print(a)