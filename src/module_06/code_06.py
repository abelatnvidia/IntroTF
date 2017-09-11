import os, glob, tensorflow as tf

# create a file pattern to pick up csv files in pwd
file_pattern = os.path.join(os.getcwd(), '*.csv')

# get list of csv files in the current directory
csv_files = glob.glob(file_pattern)

# create a queue of files
file_queue = tf.train.string_input_producer(csv_files,)

# init Reader that outputs the lines of a file delimited by newlines
reader = tf.TextLineReader(skip_header_lines=1)

# read k,v pairs from the file queue contents
key, get_record_op = reader.read(file_queue)

# A list of Tensor objects with types from: float32, int32, int64, string.
# One tensor per column of the input record, with either a scalar default
# value for that column or empty if the column is required.
column_descriptors = [[0.0], [0.0], [0.0], [0.0], ['Absent'], [0.0], [0.0], [0.0], [0.0], [0.0]]

# parse a line of data using the descriptors for each element
decode_ops = tf.decode_csv(get_record_op, record_defaults=column_descriptors)

# create a predicate
predicate = tf.equal(decode_ops[4], tf.constant('Present'))

# convert the 5th value based on present/absent to 1.0/0.0
decode_ops[4] = tf.cond(predicate, lambda: tf.constant(1.0), lambda: tf.constant(0.0))

# extract the predictor data
data_ops = tf.stack(decode_ops[0:9])

# extract the label from input
label_ops = decode_ops[9]

# configure dimensions of the data and model
param_sz = 9
label_sz = 1
batch_sz = 32

# create a batch(er) queue
data_batch_ops, label_batch_ops = \
    tf.train.shuffle_batch([data_ops, label_ops],
                           batch_size               = batch_sz,
                           capacity                 = 32*10   ,
                           min_after_dequeue        = 32      ,
                           num_threads              = 1       ,
                           seed                     = None    ,
                           enqueue_many             = False   ,
                           shapes                   = None    ,
                           allow_smaller_final_batch= False   ,
                           shared_name              = None    ,
                           name                     = None
                           )

# Since there are 9 predictor variables (i.e. CSV columns) we init models with 9 parameters
params_init = tf.random_normal(shape=[param_sz, label_sz], stddev=0.01, name='model-params-init', dtype=tf.float32)

# create variable to capture init params
w = tf.Variable(params_init, name='model-params', trainable=True)
b = tf.Variable(tf.zeros(shape=[label_sz]),name='bias')

# create a placeholder to pass in the data X values
X = tf.placeholder(tf.float32, shape=[batch_sz,param_sz], name='data')

# create place holder to pass in the Y values
Y = tf.placeholder(tf.float32, shape=[batch_sz], name='labels')

# with perfect parameters/model: records*params = labels
y = tf.add(tf.matmul(X, w), b, name='model-values')

# compute difference between model predictions and actual labels
error = tf.subtract(Y, tf.squeeze(y), name='model-error')

# calculate the loss using error
loss = tf.reduce_sum(tf.abs(error))

# init global step variable
global_step = tf.Variable(1, trainable=False, name='global-step')

# init optimizer for this loss function
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)

# compute the gradients of the loss function w.r.t. params
compute_gradient_op = optimizer.compute_gradients(loss, var_list=[w,b])

# apply the gradients to update the parameters
apply_gradient_op = optimizer.apply_gradients(
    compute_gradient_op       ,
    global_step=global_step   ,
    name       ='param-update'
)

# create some loss summary operations
loss_summary_op = tf.summary.scalar('loss',loss)

# keep that linter happy
i = 0; loss_value = 0

# init session ...
with tf.Session() as sess:

    # open summary file writer for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(), sess.graph)

    # init vars
    sess.run(tf.global_variables_initializer())

    # Start populating the filename queue.
    queue_coordinator = tf.train.Coordinator()
    queue_worker_threads = tf.train.start_queue_runners(coord=queue_coordinator)

    for i in range(100):

        # get a batch of data records and associated labels
        [data_batch, label_batch] = sess.run([data_batch_ops, label_batch_ops])

        # update model params based on this batch of data and labels
        sess.run([apply_gradient_op,loss],{X:data_batch,Y:label_batch})

        # get the latest loss value
        loss_summary,loss_value = sess.run([loss_summary_op,loss],{X:data_batch,Y:label_batch})

        #write the summary to file for tensorboard
        sfw.add_summary(loss_summary,i)

        # blab about it on stdout
        if i % 10 == 0: print('loss at iteration {} is: {:0.2f}'.format(i, loss_value))

    # close down the queue
    queue_coordinator.request_stop()
    queue_coordinator.join(queue_worker_threads)

    # clean up
    sfw.close()

# blab about final loss value
print('loss at iteration {} is: {:0.2f}'.format(i,loss_value))

'''
    So finally we get our optimizer set up and running infinite stream of csv data
'''