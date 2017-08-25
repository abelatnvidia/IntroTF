import tensorflow as tf

# create some computation to perform
z = tf.add(2,3)

# init session and compute stuff
with tf.Session() as sess: result = sess.run(z)

# blab about result
print('the result from sess.run is: {}'.format(result))

# don't forget to keep track of what time is output from session
print('the output type from run is {}'.format(type(result)))