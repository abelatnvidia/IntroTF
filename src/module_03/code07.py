import os, tensorflow as tf

# use the range operation to generate vectors of sequential values
x = tf.range(start=0,limit=10,delta=1,dtype=tf.int32,name='range')

with tf.Session() as sess:

    # init log files for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # dump the graph definition as JSON
    print(sess.graph.as_graph_def())

    # execute the range operation to generate vector
    result = sess.run(x)

    # clean up
    sfw.close()

# show the result of the computation
print('result of executing range operation: {}'.format(result))

'''
Be sure to understand that tf.range is an operation that when
executed produces a 1-dim tf.Tensor object of shape=[10].

Have a look at the graph def output and tensorboard

Notice that all the input args about the range such as start, 
limit, etc are stored as constant operators in the compute graph 
definition.  Therefore, when TF goes to evalueate the range operation
it must find all input nodes (i.e. constant operations whoes output
flows into the range operation) and execute those operations first 
to generate the necessary input to execute the range operator.

From this perspective, it should be clear why tf.range can not 
be used in the feeds_dict as input to sess.run.
'''