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
    result = sess.run(y, {x: tf.range(10)}) # ERROR

    # clean up
    sfw.close()

# blab about the result
print('The value(s) produced for y is: {}'.format(result))

'''
We extend the previous example again to show passing 
tf.Tensor to sess.run (kaboom!)

The general idea behind the feed_dict is to pass external
values into a TensorFlow compute session. 

Let's investigate tf.range a bit more ...
'''