import os, tensorflow as tf

# create a variable in default graph
a = tf.Variable(10)

with tf.Session() as sess:

    # create tensorboard log files
    sfw = tf.summary.FileWriter(os.getcwd(), sess.graph)

    # dump the graph definition as JSON
    print(sess.graph.as_graph_def())

    # init variables
    sess.run(a.initializer)

    # run something
    result = sess.run(a)

    # You must evaluate a variable in session and use the result
    print('the value of variable is: {}'.format(a.eval()))

    # then, what is difference between sess.run(a) and a.eval()??
    print('the operation result is: {}'.format(result))

    # clean up
    sfw.close()


