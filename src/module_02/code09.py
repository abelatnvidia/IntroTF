import tensorflow as tf

# create variable operations, specify initial values
x = tf.Variable(2, name='x')
y = tf.Variable(3, name='y')

# create variable operations to assign a new value to vars
xassop = x.assign(20)
yassop = y.assign(30)

# scope session for computation
with tf.Session() as sess:

    # init variables
    sess.run(x.initializer)
    sess.run(y.initializer)

    # run the x and y assign operations
    sess.run(xassop)
    sess.run(yassop)

    print(x.eval())
    print(y.eval())