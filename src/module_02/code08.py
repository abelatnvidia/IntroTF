import tensorflow as tf

# create variable operations, specify initial values
x = tf.Variable(2, name='x')
y = tf.Variable(3, name='y')

# create operations to assign a new value to vars
xassop = x.assign(20)
yassop = y.assign(30)

# scope session for computation
with tf.Session() as sess:

    # executes Assign operation with
    # constant initial value and
    # memory location associated with variable
    sess.run(x.initializer)
    sess.run(y.initializer)

    # dereference variables (x/read,y/read)
    # x.eval() returns result of executing the identity operation on x
    print(x.eval())
    print(y.eval())

    # note that assign operation has two inputs but no outputs
    # so do not try result = sess.run(x.initializer)
    # b/c result will be None since assign has no ouput

    '''
    ref: https://www.tensorflow.org/versions/r1.3/programmers_guide/graphs
    Executing v = tf.Variable(0) adds to the graph a tf.Operation that 
    will store a writeable tensor value that persists between tf.Session.run 
    calls. The tf.Variable object wraps this operation, and can be used 
    like a tensor, which will read the current value of the stored value. 
    The tf.Variable object also has methods such as assign and assign_add 
    that create tf.Operation objects that, when executed, update the stored 
    value. (See Variables for more information about variables.)
    
    '''