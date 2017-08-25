import os, tensorflow as tf

# often we might not know value of var or const
p = tf.placeholder(tf.int32,1,name='a')

with tf.Session() as sess:

    # init log files for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # dump the graph definition as JSON
    print(sess.graph.as_graph_def())

    # clean up
    sfw.close()