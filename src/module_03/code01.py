import os, tensorflow as tf

# often we might not know value of var or const
p = tf.placeholder(tf.int32,name='p')

with tf.Session() as sess:

    # init log files for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # evaluate the placeholder variable (kaboom!)
    result = sess.run(p)

    # clean up
    sfw.close()

# blab about the result
print('The value produced p is: {}'.format(result))

'''
    This should produce InvalidArgumentError since "p" does not have a value
    Since p does not have value it can not be evaluated
    Notice that p is a "Placeholder" operation
    Without a value, when p is evaluated it has nothing to output
'''