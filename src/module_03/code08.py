import os, tensorflow as tf

# create placeholder operation for limit of range
rangeLimit = tf.placeholder(tf.int32,shape=[],name='range-limit')

# use the range operation to generate vectors of sequential values
x = tf.range(start=0, limit=rangeLimit, delta=1, dtype=tf.int32, name='range')

with tf.Session() as sess:

    # init log files for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(), sess.graph)

    # dump the graph definition as JSON
    print(sess.graph.as_graph_def())

    # execute the range operation to generate vector
    result = sess.run(x,{rangeLimit:5})

    # clean up
    sfw.close()

# show the result of the computation
print('result of executing range operation: {}'.format(result))

'''
Again, when TF goes to evalueate the range operation
it must find all input nodes (i.e. operations whoes output
flows into the range operation) and execute those operations first 
to generate the necessary input to execute the range operator.

In this example, we made the limit of the tf.range a placeholder 
operation which must be defined as feed_dict to sess.run
'''