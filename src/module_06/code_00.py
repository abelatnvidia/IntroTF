import os,glob, tensorflow as tf

# create a file pattern to pick up csv files in pwd
file_pattern = os.path.join(os.getcwd(),'*.csv')

# get list of csv files in the current directory
csv_files = glob.glob(file_pattern)

# create a queue of files
file_queue = tf.train.string_input_producer(csv_files)

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

    # ok lets run the file queue size op and see what we get
    file_queue_size = sess.run(file_queue.size())

    # clean up
    sfw.close()

# finally, the file queue size (... wait what??)
print('the result of queue size operation: {}'.format(file_queue_size))