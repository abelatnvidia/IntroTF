import os, tensorflow as tf

# create an operation to generate random matrix from some distribution
rand = tf.random_uniform((2,2), minval=-1, maxval=1, dtype=tf.float32, name='random-op')

# create variables initialized to random values specified by rand operation
X = tf.Variable(rand, name='X')
Y = tf.Variable(rand, name='Y')

'''
    Objective: if x(i,j)>y(i,j) then z(i,j) = x(i,j)+y(i,j)
               if x(i,j)<y(i,j) then z(i,j) = x(i,j)-y(i,j)
               if x(i,j)=y(i,j) then z(i,j) = 42
'''

# define operation to execute for each instance of predicate x>y
T1 = tf.where(tf.greater(X ,Y),tf.add      (X,Y) , X)
T2 = tf.where(tf.less   (T1,Y),tf.subtract(T1,Y) , X)
Z  = tf.where(tf.equal  (T2,Y),tf.ones_like(X)*42,T2)

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

'''
    The thing to take away here is that we can cascade calls to tf.where
    Here we would like default value of 42 for x(i,j)==y(i,j)
    We can not do this in a single call to tf.where and therefore need
    to use intermediate variables to construct Z one predicate at a time.
    Again, consider carfully the efficiency of this type of code.
    For large tensors this might not be computationally performant and/or
    consume considerable memory if there are many predicates to handle 
    since would require many temporary variables. 
'''