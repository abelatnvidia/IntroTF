import os, tensorflow as tf

# on a particular device ...
with tf.device('cpu:0'):

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

# tweak our session configuration to show where vars and ops get put
sessionConfig = tf.ConfigProto(log_device_placement=True)

# create explicit session and keep it open
sess = tf.Session(graph=computeGraph,config=sessionConfig)

# summary file for graph nodes, vars, opts etc
summaryFileWriter = tf.summary.FileWriter(outdir, sess.graph)

# init variables
sess.run(variableInitializer)

# execute the session for fetches
addResult = sess.run(addOperation)

# clean up writer
summaryFileWriter.close()

# clean up session
sess.close()

# blab about where to look for logs and events
print('logs and events dir: {}'.format(outdir))

# ops are sooo kool
print(addResult)