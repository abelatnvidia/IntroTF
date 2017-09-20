import os, tensorflow as tf

# the full file path for TF Record file with single record
record_file_path = os.path.join(os.getcwd(),'dat.tfrecord')

# init a file queue
file_queue = tf.train.string_input_producer([record_file_path],name='file_queue',num_epochs=1)

# init record reader
reader = tf.TFRecordReader()

# read the record file
_, tfrecord_read_op = reader.read(file_queue)

# parse bytes into dict
tfrecord = tf.parse_single_example(
    tfrecord_read_op,
    features={
        'image/height'     : tf.FixedLenFeature([], tf.int64 ),
        'image/width'      : tf.FixedLenFeature([], tf.int64 ),
        'image/colorspace' : tf.FixedLenFeature([], tf.string),
        'image/channels'   : tf.FixedLenFeature([], tf.int64 ),
        'image/class/label': tf.FixedLenFeature([], tf.int64 ),
        'image/class/text' : tf.FixedLenFeature([], tf.string),
        'image/format'     : tf.FixedLenFeature([], tf.string),
        'image/filename'   : tf.FixedLenFeature([], tf.string),
        'image/encoded'    : tf.FixedLenFeature([], tf.string)
    },
    name='features'
)

# image was saved as uint8, so we have to decode tf.string as uint8.
imageT  = tf.decode_raw(tfrecord['image/encoded'], tf.uint8)

# since exported as tf.int64 there is no need for tf.decode_raw
heightT = tfrecord['image/height']
widthT  = tfrecord['image/width' ]

# remember, it's all just ops, have to run to get result
with tf.Session() as sess:

    # init vars
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer ())

    # init summary file writer for tensorboard
    sfw = tf.summary.FileWriter(os.getcwd(),graph=sess.graph)

    # Start populating the filename queue.
    queue_coordinator = tf.train.Coordinator()
    queue_worker_threads = tf.train.start_queue_runners(coord=queue_coordinator)

    # kaboom!
    height = sess.run(heightT)
    width  = sess.run(widthT )

    # close down the queue
    queue_coordinator.request_stop()
    queue_coordinator.join(queue_worker_threads)

    # clean up summary file writer
    sfw.close()

# blab about it ...
print('image height x width : {}x{}'.format(height,width))

'''
    This example throws an exception that the file_queue is empty (!)
    The exception is triggered by multiple calls for height and width.
    
    With this current configuration, must query all properties in a 
    single call to sess.run() otherwise, each sess.run() will invoke
    additional file parse.  
    
    The better design is to use a placeholder for the serialized TF Record
    This way, when we work with the placeholder, we make it explicit 
    which record we are working with and mitigate the chances that 
    additional file is parsed for each record.
    
    Notice that if there were many record files that, for this example,
    the height would come from file one but the width would (inadvertently)
    come from the second file 2 and so on.
    
    The issue becomes clear since num_epochs=1 but what is num_epochs=None
    in which the file queue will serve the single file infinitely?  In such
    a siutation, TF would re-parse the TFRecord every time a property was
    evaluated -- very poor performance (don't do this!).
    
    
'''