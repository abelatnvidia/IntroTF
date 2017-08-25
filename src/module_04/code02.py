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
a = tf.Variable(1.0 ,name='slope'    ,dtype=tf.float32)
b = tf.Variable(16.0,name='intercept',dtype=tf.float32)

# create summary operations to track the parameter values
a_sum_op = tf.summary.scalar('slope'    ,a)
b_sum_op = tf.summary.scalar('intercept',b)

# compute model values: a*X+b
y = tf.add(tf.multiply(a,X),b)

# compuet the model error
error = tf.subtract(Y,y,name='error')

# compute the loss function
loss = tf.square(error,name='loss')

with tf.Session() as sess:

    # init summary file writer for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # init model parameters variables
    sess.run(a.initializer)
    sess.run(b.initializer)

    # compute the loss
    result = sess.run(loss,{X:data[:,0],Y:data[:,1]})

    # execute summary operations
    a_sum = sess.run(a_sum_op)
    b_sum = sess.run(b_sum_op)

    # write parameter summaries to log file for tensorboard
    sfw.add_summary(a_sum)
    sfw.add_summary(b_sum)

    # clean up
    sfw.close()

print(result)

'''
    Tracking additional parameters for tensorboard is easy.
    Just create tf.summary.scalar operations and write 
    the result of the summary operations to the summary
    file writer. You can imagine that with many summary
    operations it could get a little tideous to execute
    every summary operation explicitly.  TF offers a way
    to "merge" all summary operations into a single summary
    operation whose result contains all summaries which
    can then be written as a single call to the summary
    file writer.
'''