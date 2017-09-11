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
decode_record_ops = tf.decode_csv(get_record_op, record_defaults=column_descriptors)

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
    Here we initialize a queue of files which are then provided to a reader.
    The tf.TextLineReader produces "records" which need to be decoded
    The tf.decode_csv is used to parse the records and provide usable data
    
    Don't forget, everything is an operation!  Therefore, once in session,
    we must execute the decode_record operation, which depends on the 
    get_record operation, which depends on the string_input_producer
    
    Keep in mind, we organized the operations this way to create this
    compute graph of decode --> fetch --> reader --> file_queue
    
    Holy cow, this is getting complicated for just importing CSV files!
    But we have to realize that this is an asynchronous and highly scalable
    implementation that should scale to many many many data.  
    
    So that is the trade-off, simple code that doesn't scale or 
    complicated code that does.
    
'''