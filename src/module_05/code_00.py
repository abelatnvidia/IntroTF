import os, tensorflow as tf

# create an operation to generate random data from some distribution
X = tf.random_uniform([3,3],minval=-1,maxval=1,dtype=tf.float32,name='X')

# init computation session
with tf.Session() as sess:

    # init tensorboard logs
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # dump the graph definition as JSON
    print(sess.graph.as_graph_def())

    # execute the op to get result
    xResult1 = sess.run(X)

    # if execute same op again, get different data
    xResult2 = sess.run(X)

    # clean up
    sfw.close()

# blab about the results
print('x Result 1:\n{}'.format(xResult1))
print('x Result 2:\n{}'.format(xResult2))

'''
    It might be supprising to see all the output of the graph definition (!)
    Recall that to get uniform data on the interval [a,b] must compute: 
    a + (b-a)*uniform(sz) where uniform(sz) produces uniform random data on 
    the unit interval [0,1] of size sz.  Therefore, the op tf.random_uniform 
    is actually a compute graph for the aforementioned transformation.
    
    Don't forget there are no variables, per se, only operations that when executed 
    produce a result.  The point here is that if evaluate the X operation we get some 
    random data as a result of the associated operation.  If evaluate X operation
    again then get a different random result. 
    
    It should also be clear that X is *not* even a tf.Variable (!)
    Notice that we did not have to execute initializer on X.
    This reinforces the idea that X is a subgraph of operations with some initial 
    conditions (minval, maxval, uniform)that when executed produce random result.
    
'''

