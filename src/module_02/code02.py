import os, tensorflow as tf

# create a variable in default graph
a = tf.Variable(10)

with tf.Session() as sess:

    # create tensorboard log files
    sfw = tf.summary.FileWriter(os.getcwd(), sess.graph)

    # dump the graph definition as JSON
    print(sess.graph.as_graph_def())

    # clean up
    sfw.close()