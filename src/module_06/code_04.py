import os, glob, tensorflow as tf

# create a file pattern to pick up csv files in pwd
file_pattern = os.path.join(os.getcwd(),'*.csv')

# get list of csv files in the current directory
csv_files = glob.glob(file_pattern)

# create a queue of files
file_queue = tf.train.string_input_producer(csv_files)

# init Reader that outputs the lines of a file delimited by newlines
reader = tf.TextLineReader(skip_header_lines=1)

# read k,v pairs from the file queue contents
key, get_record_op = reader.read(file_queue)

# A list of Tensor objects with types from: float32, int32, int64, string.
# One tensor per column of the input record, with either a scalar default
# value for that column or empty if the column is required.
column_descriptors = [[0.0],[0.0],[0.0],[0.0],['Absent'],[0.0],[0.0],[0.0],[0.0],[0]]

# parse a line of data using the descriptors for each element
decode_ops = tf.decode_csv(get_record_op, record_defaults=column_descriptors)

# create a predicate
predicate = tf.equal(decode_ops[4], tf.constant('Present'))

# convert the 5th value based on present/absent to 1.0/0.0
decode_ops[4] = tf.cond(predicate, lambda: tf.constant(1.0), lambda: tf.constant(0.0))

# extract the predictor data
data_ops = decode_ops[0:9]

# extract the label from input
label_ops = decode_ops[9]

# create a batch(er) queue
data_batch_ops, label_batch_ops \
    = tf.train.shuffle_batch([data_ops, label_ops],
                             batch_size               = 2    ,
                             capacity                 = 32*10,
                             min_after_dequeue        = 32   ,
                             num_threads              = 1    ,
                             seed                     = None ,
                             enqueue_many             = False,
                             shapes                   = None ,
                             allow_smaller_final_batch= False,
                             shared_name              = None ,
                             name                     = None
                             )

# init session ...
with tf.Session() as sess:

    # open summary file writer for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # Start populating the filename queue.
    queue_coordinator    = tf.train.Coordinator()
    queue_worker_threads = tf.train.start_queue_runners(coord=queue_coordinator)

    # get a single row of data
    [data, labels] = sess.run([data_batch_ops, label_batch_ops])

    # close down the queue
    queue_coordinator.request_stop()
    queue_coordinator.join(queue_worker_threads)

    # clean up
    sfw.close()

# ok, print the batch of data from the csv file
print('got a batch of data [{},{}]: \n{}'.format(data.shape[0],data.shape[1],data))

# print the associated labels for this batch
print('got a batch of labels for the data: {}'.format(labels))

'''
    First we have separated out the operations provided by the decoder
    into operations that produce data and operations that produce labels
    
    Using these data and label operations we can start to organize 
    labeled examples for model training pourposes. 

    In this example, we've added a shuffle_batch queue after the record decoder
    This shuffle-batch queue will get filled up with collections of decoded
    records which can be dequeue and used as a batch of data during training.
    
    There are many default input arguments to the shuffle batch constructor.
    Before thinking about what each of the args do remember that queues are
    asynchronous and have their own worker threads. Typically, the input args
    configure how data can be insterted, how data can be obtained, and 
    how much work the threads can do asynchronously.  That is, large queue 
    capacity means that the worker threads can create many many batches in 
    advance -- but this takes lots of memory and compute resources to do. 
    There may be many queues in your design all which have their own threads
    that need to do some work asynchronously.  If there is one queue that
    has huge capacity and many workers, it could gobble up all the system
    resources and starve the other queues from performing well.  
'''