import os, tensorflow as tf

# create placeholder op
x = tf.placeholder(tf.float32,shape=[],name='x')

# init two variable ops to produce model values
a = tf.Variable(3.14,name='slope'    )
b = tf.Variable(2.71,name='intercept')

# construct operation to compute model
y = a*x + b

with tf.Session() as sess:

    # init log files for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # don't forget to initialize your variables!
    sess.run(a.initializer)
    sess.run(b.initializer)

    # evaluate the model operation for different values of x
    result = sess.run(y,{x:2})

    # clean up
    sfw.close()

# blab about the result
print('The value produced for y is: {}'.format(result))

'''
    In this example, we create a simple linear model f(x) = ax + b
    where we use tf.Variable operations to hold parameters {a,b} and 
    tf.Placeholder to act as the free variable.  
'''