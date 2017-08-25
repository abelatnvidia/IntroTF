import os, tensorflow as tf

# these are variables
x = tf.Variable(2, name='x')
y = tf.Variable(3, name='y')

# this is an operation not a variable
addOperation = tf.add(x,y, name='add')

# where to output logs and events
outdir = os.getcwd()

# scope
with tf.Session() as sess:

    # summarize the graph
    writer = tf.summary.FileWriter(outdir,sess.graph)

    # init variables
    sess.run(x.initializer)
    sess.run(y.initializer)

    # execute the session for fetches
    addResult = sess.run(addOperation)

    # clean up
    writer.close()

# blab about where to look for logs and events
print('logs and events dir: {}'.format(outdir))

# ops are sooo kool
print(addResult)