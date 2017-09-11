import os, glob, time, tensorflow as tf

# create a file pattern to pick up csv files in pwd
file_pattern = os.path.join(os.getcwd(),'*.csv')

# get list of csv files in the current directory
csv_files = glob.glob(file_pattern)

# create a queue of files [default args assigned]
file_queue = tf.train.string_input_producer(
                                             csv_files        ,
                                             num_epochs = None,
                                             shuffle    = True,
                                             seed       = None,
                                             capacity   = 32  ,
                                             shared_name= None,
                                             name       = None,
                                             cancel_op  = None
                                            )
# blab about what we're looking for
print('locating data input files: {}'.format(file_pattern))

# print the files located
for f in csv_files: print('found data input file: {}'.format(f))

# do you know what type of operation this is?
print('type of file queue is: {}'.format(type(file_queue)))

# print the size of the queue (gotcha: remember everything is an op!)
print('the size of the queue is: {}'.format(file_queue.size()))

# init session ...
with tf.Session() as sess:

    # open summary file writer for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),sess.graph)

    # Start populating the filename queue.
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    # wait for queue runners to init string_input_producer FIFO queue
    time.sleep(0.1)

    # ok lets run the file queue size op and see what we get
    file_queue_size = sess.run(file_queue.size())

    # close down the queue
    coord.request_stop()
    coord.join(threads)

    # clean up
    sfw.close()

# finally, the file queue size (that looks better ... wait what??)
print('the result of queue size operation: {}'.format(file_queue_size))

'''
    When using Queues we must create tf.train.Coordinator and 
    manually start extra "queue runner" threads which actually
    populate the queue.  Once we're done with the queue we need 
    to (manually) clean up the threads using the coordinator
    
    Since the queue runners are separate threads we must add
    a short sleep before asking for size()
    
    Notice that the size returned was actually the capacity of
    the queue.  How then do we figure how many files/resources
    were provided to the string_input_producer?
'''