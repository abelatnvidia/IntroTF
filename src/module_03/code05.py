import os, tensorflow as tf, numpy as np

# create placeholder op (NOTE shape now vector)
x = tf.placeholder(tf.float32, shape=[10], name='x')

# init two variable ops to produce model values
a = tf.Variable(3.14, name='slope')
b = tf.Variable(2.71, name='intercept')

# construct operation to compute model
y = a * x + b

with tf.Session() as sess:
    # init log files for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(), sess.graph)

    # don't forget to initialize your variables!
    sess.run(a.initializer)
    sess.run(b.initializer)

    # evaluate the model operation for different values of x
    result = sess.run(y, {x: np.arange(10)})

    # clean up
    sfw.close()

# blab about the result
print('The value(s) produced for y is: {}'.format(result))

'''
We extend the previous example to show passing numpy array (!)
    '''