import os, tensorflow as tf

# these are variables
x = tf.Variable(2, name='x')
y = tf.Variable(3, name='y')

# init operation for all global variables
globalVariableInitializer = tf.global_variables_initializer()

# this is an operation not a variable
addOperation = tf.add(x, y, name='addop')

# where to output logs and events
outdir = os.getcwd()

# scope
with tf.Session() as sess:

    # summary file for graph events
    summaryFileWriter = tf.summary.FileWriter(outdir, sess.graph)

    # init variables
    sess.run(globalVariableInitializer)

    # execute the session for fetches
    addResult = sess.run(addOperation)

    # clean up
    summaryFileWriter.close()

# blab about where to look for logs and events
print('logs and events dir: {}'.format(outdir))

# ops are sooo kool
print(addResult)