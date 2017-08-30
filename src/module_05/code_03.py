import os, tensorflow as tf

# create an operation to generate random number from some distribution
rand = tf.random_uniform((), minval=-1, maxval=1, dtype=tf.float32, name='random-op')

# create variables initialized to random values specified by rand operation
X = tf.Variable(rand, name='X')
Y = tf.Variable(rand, name='Y')

'''
    Objective: if x> y then z = x+y
               if x<=y then z = x-y
'''

# define operation to execute for each instance of predicate x>y
Z = tf.cond(
            tf.greater (X, Y),
    lambda: tf.add     (X, Y),
    lambda: tf.subtract(X, Y),
    name='Z'
)

# init computation session
with tf.Session() as sess:

    # init tensorboard logs
    sfw = tf.summary.FileWriter(os.getcwd(), sess.graph)

    # init variables
    sess.run(X.initializer)
    sess.run(Y.initializer)

    # execute the X op to get values
    valX = sess.run(X)

    # execute the Y op to get the Y values
    valY = sess.run(Y)

    # execute Z operation to form combination of X,Y
    valZ = sess.run(Z)

    # clean up
    sfw.close()

# blab about the results
print('value of X:\n{}'.format(valX))
print('value of Y:\n{}'.format(valY))
print('value of Z:\n{}'.format(valZ))

# just to see both results for ourselves
print('value of X+Y:\n{}'.format(valX+valY))
print('value of X-Y:\n{}'.format(valX-valY))

# just to really nail this sucker down ...
if valX>valY:
    assert valZ == valX+valY
else:
    assert valZ == valX-valY

'''
    Finally, initializing tf.Variables which depend on rand-op captures
    a random value which can then be reused over and over in subsequent
    calulations without actually changing. This is b/c everytime we
    evaluate the X or Y variables we're actually evaluating X/read op and
    the Y/read op which returns whatever value is currently assinged to 
    the variable. The value assigned to the variable in this case is 
    determined during the call sess.run(X.initializer) which looks at 
    the input arg which points at the random operation.  Once the random
    operation is evaluated the result is passed back to the initializer
    which then executes the assign operation to write that value to the
    memory slot for the variable.  Whenever the variable operation is 
    evaluated, whatever value is in the memory slot is returned.
'''