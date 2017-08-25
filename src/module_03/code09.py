import os, tensorflow as tf, numpy as np

# init placeholder ops for function values
a = tf.placeholder(tf.float32,shape=[3],name='a')
b = tf.placeholder(tf.float32,shape=[3],name='b')
c = tf.placeholder(tf.float32,shape=[3],name='c')
d = tf.placeholder(tf.float32,shape=[3],name='d')

e = tf.placeholder(tf.float32,shape=[5,1],name='exponent')

# the value to compute ((a+b)^e+c)/d
y = tf.div(
        tf.add(
            tf.pow(
                tf.add(a,b),
                e
            ),
            c
        ),
    d
)

with tf.Session() as sess:

    # init logs for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # evaluate y
    result = sess.run(y,{a:[2,2,2],b:[3,3,3],c:[4,4,4],d:[5,5,5],e:[[1],[2],[3],[4],[5]]})

    # clean up
    sfw.close()

print('The result of computing y is: \n{}'.format(result))

'''
We can pass in as many placehold values as we need
Be sure to check out the associated compute graph 
in the tensorboard 
'''