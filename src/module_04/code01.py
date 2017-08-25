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
a = tf.Variable(1.0,name='slope'     ,dtype=tf.float32)
b = tf.Variable(16.0,name='intercept',dtype=tf.float32)

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

    # clean up
    sfw.close()

print(result)

'''
    Notice we create placeholders for both X and Y to pass data into session.
    Via the compute graph, TF knows to compute loss, must compute error.
    To compute error, must compute model values y.
    To compute model values y, must compute parameter values a and b
    Since a and b have no additional inputs the process terminates and the
    computations are rolled back up to finally compute the loss. 
'''