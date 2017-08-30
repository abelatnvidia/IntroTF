import os, tensorflow as tf

# create an operation to generate random data from some distribution
rand = tf.random_uniform((),minval=-1,maxval=1,dtype=tf.float32,name='random-op')

'''
    Objective: if x>y then z = x+y
               if x<y then z = x-y
'''

# define operation to execute for each instance of predicate x>y
Z = tf.cond(
                tf.greater (rand,rand),
        lambda: tf.add     (rand,rand),
        lambda: tf.subtract(rand,rand),
        name='Z'
)

# init computation session
with tf.Session() as sess:

    # init tensorboard logs
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # execute the X op to get values
    valX = sess.run(rand)

    # execute the Y op to get the Y values
    valY = sess.run(rand)

    # execute Z operation to form combination of X,Y
    valZ = sess.run(Z)

    # clean up
    sfw.close()

# blab about the results
print('value of X:\n{}'.format(valX))
print('value of Y:\n{}'.format(valY))
print('value of Z:\n{}'.format(valZ))

'''
    It is tempting to think that the output here will be same as previous code
    However, it is important to understand that there is only a single operation
    that is passed to both left and right inputs of tf.greater.  Therefore the 
    two input args will always be equal and thus be subtracted (i.e. call tf.subtract).
    However, again, keep in mind that the left and right operands to tf.subtract
    are not the same values provided to tf.greater.  First tf.greater operation
    is evaluated which has left and right inputs which both point to the rand operation
    Once the rand operation is evaluated the result is passed as the left and right 
    input to tf.greater.  Since tf.greater will return false in this case, the 
    tf.subtract operation will then be executed.  The tf.subtract operation takes 
    a left and a right argument where both these input args are the output of the rand
    operation.  So then the rand operation is evaluated and the output is passed to 
    tf.subtract as both arg1 and arg2.  The output of tf.subtract for identical inputs
    is always zero. 
    
    To see this, be sure to check out the associated compute graph via tensorboard
'''