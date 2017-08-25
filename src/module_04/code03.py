import os, xlrd, numpy as np, tensorflow as tf

# opent the xls file for reading
xlsfile = xlrd.open_workbook('fire_theft.xls', encoding_override='utf-8')

# there can be many sheets in xls document
sheet = xlsfile.sheet_by_index(0)

# ask the sheet for each row of data explicitly
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])

# compute the number of samples
num_samples = data.shape[0]

# create a placeholder to pass in the data X values
X = tf.placeholder(tf.float32,shape=num_samples,name='num-fire')

# create place holder to pass in the Y values
Y = tf.placeholder(tf.float32,shape=num_samples,name='num-theft')

# create tf variables for the model to be estimated
a = tf.Variable(0.0,name='slope'    ,dtype=tf.float32)
b = tf.Variable(0.0,name='intercept',dtype=tf.float32)

# create summary operations to track the parameter values
a_sum_op = tf.summary.scalar('slope'    ,a)
b_sum_op = tf.summary.scalar('intercept',b)

# operation to compute model values: a*X+b
y = tf.add(tf.multiply(a,X),b)

# operation to compute the model error w.r.t. to data
error = tf.subtract(Y,y,name='error')

# operation compute the RMSE loss function based on model error
loss = tf.sqrt(tf.reduce_mean(tf.square(error,name='loss')))

# init gradient decent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# operation to compute gradients of the loss function w.r.t. model parameters
compute_grad_op = optimizer.compute_gradients(loss=loss,var_list=[a,b])

# operation to apply gradients to model parameters
apply_grad_op = optimizer.apply_gradients(compute_grad_op,name='parameter-update')

# init session and compute stuff ...
with tf.Session() as sess:

    # init summary file writer for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # init model parameters variables
    sess.run(a.initializer)
    sess.run(b.initializer)

    # execute summary operations
    a_sum = sess.run(a_sum_op)
    b_sum = sess.run(b_sum_op)

    # write parameter summaries to log file for tensorboard
    sfw.add_summary(a_sum)
    sfw.add_summary(b_sum)

    # compute gradients and update model parameters
    sess.run(apply_grad_op,{X:data[:,0],Y:data[:,1]})

    # compute the loss
    result = sess.run(loss,{X:data[:,0],Y:data[:,1]})

    # clean up
    sfw.close()

print(result)

'''
    Parameter Updates:
    First, need to init an optimizer 
    Second, compute derivatives of the loss func w.r.t. model params
    Third, using the partial derivaties and optimizer, update parameters
'''