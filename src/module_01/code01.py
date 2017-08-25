import os, tensorflow as tf

x = tf.constant(2  , name='x')
y = tf.constant(3  , name='y')
z = tf.add     (x,y, name='z')

# where to output logs and events
outdir = os.getcwd()

# scope
with tf.Session() as sess:

    # summarize the graph
    writer = tf.summary.FileWriter(outdir,sess.graph)

    # execute the session for fetches
    result = sess.run(z)

    # clean up
    writer.close()

# blab about where to look for logs and events
print('logs and events dir: {}'.format(outdir))

# ops are sooo kool
print(result)