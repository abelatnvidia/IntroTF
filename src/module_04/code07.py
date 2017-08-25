import os, xlrd, numpy as np, tensorflow as tf, matplotlib.pyplot as plt

# opent the xls file for reading
xlsfile = xlrd.open_workbook('fire_theft.xls', encoding_override='utf-8')

# there can be many sheets in xls document
sheet = xlsfile.sheet_by_index(0)

# ask the sheet for each row of data explicitly
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])

# compute the number of samples
num_samples = data.shape[0]

# create a placeholder to pass in the data X values
X = tf.placeholder(tf.float32, shape=num_samples, name='num-fire')

# create place holder to pass in the Y values
Y = tf.placeholder(tf.float32, shape=num_samples, name='num-theft')

# compute the mean of the y-values for init intercept guess
y_mean = np.mean(data[:,1])

# create tf variables for the model to be estimated
a = tf.Variable(0.0, name='slope'    , dtype=tf.float32)
b = tf.Variable(0.0, name='intercept', dtype=tf.float32)

# operation to compute model values: a*X+b
y = tf.add(tf.multiply(a, X), b)

# operation to compute the model error w.r.t. to data
error = tf.subtract(Y, y, name='error')

# operation compute the RMSE loss function based on model error
loss = tf.reduce_mean(tf.abs(error),name='L1-loss')

# init gradient decent optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# operation to compute gradients of the loss function w.r.t. model parameters
compute_grad_op = optimizer.compute_gradients(loss=loss, var_list=[a, b])

# operation to apply gradients to model parameters
apply_grad_op = optimizer.apply_gradients(compute_grad_op, name='parameter-update')

# create summary operations to track the parameter values
a_sum_op = tf.summary.scalar('slope'    , a   )
b_sum_op = tf.summary.scalar('intercept', b   )
l_sum_op = tf.summary.scalar('loss'     , loss)

# create a merged summary operation for all summary ops
summary_op = tf.summary.merge([a_sum_op,b_sum_op,l_sum_op])

# keep the linter happy
i = 0; loss_value = 0

# init session and compute stuff ...
with tf.Session() as sess:

    # init summary file writer for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(), sess.graph)

    # init model parameters variables
    sess.run(a.initializer)
    sess.run(b.initializer)

    for i in range(3000):

        # (re)compute gradients and update model parameters for this iteration
        sess.run([apply_grad_op,loss], {X: data[:, 0], Y: data[:, 1]})

        # execute summary op for this iteration and get the latest loss value
        iter_summary,loss_value = sess.run([summary_op,loss], {X: data[:, 0], Y: data[:, 1]})

        # write loss summary for this iteration to file write
        sfw.add_summary(iter_summary,i)

        # blab about it on stdout
        if i%1000 == 0: print('loss at iteration {} is: {:0.2f}'.format(i,loss_value))

    # get the final model values
    slope,intercept = sess.run([a,b],{X: data[:, 0], Y: data[:, 1]})

    # clean up
    sfw.close()

# blab about final loss value
print('loss at iteration {} is: {:0.2f}'.format(i,loss_value))

# blab about the solution
print('the model values are: a={:0.3f}, b={:0.3f}'.format(slope,intercept))

# assign for readability
xdat = data[:,0]; ydat = data[:,1]

# plot the original data
plt.plot(xdat,ydat,'o',label='data', markersize=3)

# add the estimated linear model
plt.plot(xdat, slope*xdat + intercept, 'r', label='model')

# configure the plot
plt.legend(); plt.grid(True)

# we're done
plt.show()

'''
    Since we're not using linear method no need for L2 loss function (!!)
    Lets try out an L1 loss function and see how our model changes
'''