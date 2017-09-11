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
decode_record_ops = tf.decode_csv(get_record_op, record_defaults=column_descriptors,field_delim=',')

# create a predicate (i.e. boolean test)
predicate = tf.equal(decode_record_ops[4], tf.constant('Present'))

# convert the 5th value based on present/absent to 1.0/0.0
decode_record_ops[4] = tf.cond(predicate, lambda: tf.constant(1.0), lambda: tf.constant(0.0))

# init session ...
with tf.Session() as sess:

    # open summary file writer for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # Start populating the filename queue.
    queue_coordinator    = tf.train.Coordinator()
    queue_worker_threads = tf.train.start_queue_runners(coord=queue_coordinator)

    # get a single row of data
    record = sess.run(decode_record_ops)

    # close down the queue
    queue_coordinator.request_stop()
    queue_coordinator.join(queue_worker_threads)

    # clean up
    sfw.close()

# ok, print the line of data from the csv file
print('got a line of data: {}'.format(record))

'''
    See here that we actually get back a tensor/collection of ops
    when for the decode process (one for each column of data in CSV).
    In the decode process, sess.run(decode_ops) actually calls each
    operation one-by-one and then consolodates the op results back
    into a tensor with the appropriate order. 
    
    So the line of string data (i.e. the record) produced by the 
    TextLineReader is choped up by the decoder into parts using 
    the ',' delimiter.  Each of these parts is given to its repective
    decoder operation.  If we used a different delimiter say ";" 
    instead of ",", then we could for example have a field that 
    contained values of a vector, or matrix: 
    
                     1,2,3,4; 5,6,7,8; 9,10,11,12 
                     
    could be decoded into 3 elements with dimension 1x4 or 2x2
    
    The interesting thing is that you can take the decode operation
    and add additional operations such as convert strings to values etc
    That is, the decoder provides a set of default operations which
    you can use just like any other operation.  This allows for simple
    creation of data pre-processing compute graphs
'''