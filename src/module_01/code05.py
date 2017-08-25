import os, tensorflow as tf

# initialize an empty compute graph
computeGraph = tf.Graph()

# add variables and operations to our graph
with computeGraph.as_default():

    # these are variables
    x = tf.Variable(2, name='x')
    y = tf.Variable(3, name='y')

    # this is an operation not a variable
    addOperation = tf.add(x, y, name='addop')

    # variable init operation
    variableInitializer = tf.variables_initializer([x,y],name='varinit')

# where to output logs and events
outdir = os.getcwd()

# scope
with tf.Session(graph=computeGraph) as sess:

    # summary file for graph events
    summaryFileWriter = tf.summary.FileWriter(outdir, sess.graph)

    # init variables
    sess.run(variableInitializer)

    # execute the session for fetches
    addResult = sess.run(addOperation)

    # clean up
    summaryFileWriter.close()

# blab about where to look for logs and events
print('logs and events dir: {}'.format(outdir))

# ops are sooo kool
print(addResult)