import os, tensorflow as tf

# create placeholder op (NOTE shape now vector)
x = tf.placeholder(tf.float32,shape=[10],name='x')

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
    result = sess.run(y,{x:[0,1,2,3,4,5,6,7,8,9]})

    # clean up
    sfw.close()

# blab about the result
print('The value(s) produced for y is: {}'.format(result))

'''
    We extend the previous example here to demonstrate that
    operators are just as happy to operate on vectors as they
    are on scalars.  Likewise, ops will work on matricies too.
    In general, ops can work with any n-dim tensor input/output.
    
    In TensorFlow, 
    
        a scalar is a 0-dimensional tensor
        a vector is a 1-dimensional tensor
        a matrix is a 2-dimensional tensor
        ...
        and so on
        
        So you can think of this as a bunch of tensors being passed 
        back and forth between operations.  If one were feeling
        romantic, you might say the tensors "flow" between operators
        in a compute graph.
'''