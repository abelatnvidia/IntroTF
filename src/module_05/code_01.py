import os, tensorflow as tf

# create an operation to generate random data from some distribution
X = tf.random_uniform((),minval=-1,maxval=1,dtype=tf.float32,name='X')
Y = tf.random_uniform((),minval=-1,maxval=1,dtype=tf.float32,name='Y')

'''
    Objective: if x>y then z = x+y
               if x<y then z = x-y
'''

# define two functions one for each case
def true_func (): return tf.add     (X,Y)
def false_func(): return tf.subtract(X,Y)

# define operation to execute for each instance of predicate x(i)>y(i)
Z = tf.cond(tf.greater(X,Y),true_fn=true_func,false_fn=false_func,name='Z')

# init computation session
with tf.Session() as sess:

    # init tensorboard logs
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # execute the X op to get values
    valX = sess.run(X)

    # execute the Y op to get the Y values
    valY = sess.run(Y)

    # define operation to execute for each instance of predicate x(i)>y(i)
    #Z = tf.cond(valX > valY, true_fn=true_func, false_fn=false_func, name='Z')

    # execute Z operation to form combination of X,Y
    valZ = sess.run(Z)

    # clean up
    sfw.close()

# blab about the results
print('value of X:\n{}'.format(valX))
print('value of Y:\n{}'.format(valY))
print('value of Z:\n{}'.format(valZ))

'''
    Check carefully, notice that the value of z printed is not consistent with 
    the values of X and Y.  Why?
    
    When Z is evaluated, the predicate (tf.greater) depends on operations x and y
    which are then evaluated to produce the uniform random values on the interval.
    
    The values of X and Y produced when evaluating the predicate are completely 
    different than the values of X and Y produced when we explicitly evaluate
    those operations (i.e. sess.run(X) and sess.run(Y)).  It is a little tricky at
    first to stop thinking about variables and start thinking about op results (!!)
    
    Notice that the first argument to tf.random_uniform is empty shape which in turn
    will produce a 0-dimensional value (i.e. a scalar).  Recall that 0-d is scalar,
    1-d is vector, 2-d is matrix, 3-d is tensor, and anything larger is just "N-d".
    
'''