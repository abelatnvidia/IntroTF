import os, tensorflow as tf

# define operation to create variable subgraph
x = tf.Variable(7,name='x')

# define two different assign operations on variable
xassop1 = x.assign(77 )
xassop2 = x.assign(777)

# create two different compute sessions explicitly
sess1 = tf.Session()
sess2 = tf.Session()

# execute var op and assign op in first compute session
with sess1.as_default():
    sess1.run(x.initializer)
    result = sess1.run(xassop1)
    print('x variable value in session 1 is: {}'.format(result))

# execute var op and assign operations in the second session
with sess2.as_default():
    sess2.run(x.initializer)
    result = sess2.run(xassop2)
    print('x variable value in session 2 is: {}'.format(result))

# notice that we can ask for value of x/read according to session one
with sess1.as_default():
    print('x variable value in session 1 is: {}'.format(x.eval()))

# we can evauate x/read identity operation in session two also
with sess2.as_default():
    print('x variable value in session 2 is: {}'.format(x.eval()))