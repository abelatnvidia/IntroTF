import os, tensorflow as tf

# often we might not know value of var or const
p = tf.placeholder(tf.int32,shape=[],name='p')

with tf.Session() as sess:

    # init log files for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # evaluate the placeholder variable with feeds=dict()
    result = sess.run(p,{p:42})

    # clean up
    sfw.close()

# blab about the result
print('The value produced by p is: {}'.format(result))

'''
    Notice that we have to pass in dictionary with value of p to sess.run
    This is very common pattern for many workflows
    For example, this is how to provide image tensor to CNN for inference
    
    Finally, take note that we didn't have to execute an initializer for p
    
    See ref: https://www.tensorflow.org/api_docs/python/tf/placeholder
'''