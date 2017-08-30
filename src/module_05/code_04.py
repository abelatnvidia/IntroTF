import os, tensorflow as tf

# create an operation to generate random matrix from some distribution
rand = tf.random_uniform((2,2), minval=-1, maxval=1, dtype=tf.float32, name='random-op')

# create variables initialized to random values specified by rand operation
X = tf.Variable(rand, name='X')
Y = tf.Variable(rand, name='Y')

'''
    Objective: if x(i,j)> y(i,j) then z(i,j) = x(i,j)+y(i,j)
               if x(i,j)<=y(i,j) then z(i,j) = x(i,j)-y(i,j)
'''

# define operation to execute for each instance of predicate x>y
Z = tf.where(tf.greater(X,Y),tf.add(X,Y),tf.subtract(X,Y))

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
    The tf.cond operation only works with 0-d data 
    If we want to do more complex operations on N-d data we need to 
    use other operations such as tf.where().
    
    Notice that all the computation is done for all three operations 
    and then then result is composed.  Just keep this in mind as 
    the tensors grow quite large.  
'''