import os, tensorflow as tf

# create a constant in default graph
a = tf.constant(10)

with tf.Session() as sess:

    # create tensorboard log files
    sfw = tf.summary.FileWriter(os.getcwd(), sess.graph)

    # dump the graph definition as JSON
    print(sess.graph.as_graph_def())

    # clean up
    sfw.close()

# notice that the constant is actual listed in output as an "op" (?)