import tensorflow as tf

# init variables
x = tf.Variable(2, name='x')
y = tf.Variable(3, name='y')

# set the values of the variables
x.assign(20)
y.assign(30)

# scope session for computation
with tf.Session() as sess:

    # init variables
    sess.run(x.initializer)
    sess.run(y.initializer)

    # print the value of variables
    print(x.eval())
    print(y.eval())