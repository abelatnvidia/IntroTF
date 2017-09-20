import os, tensorflow as tf

# the full file path for TF Record file with single record
record_file_path = os.path.join(os.getcwd(),'dat.tfrecord')

# init a file queue (i know, i know, it's *just* one file ...)
file_queue = tf.train.string_input_producer([record_file_path],name='file_queue',num_epochs=1)

# init record reader
reader = tf.TFRecordReader()

# read the record file
_, tfrecord_read_op = reader.read(file_queue)

# parse serialized byte stream into dict
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

    # init summary file writer
    sfw = tf.summary.FileWriter(os.getcwd(),graph=sess.graph)

    # Start populating the filename queue.
    queue_coordinator = tf.train.Coordinator()
    queue_worker_threads = tf.train.start_queue_runners(coord=queue_coordinator)

    height, width = sess.run([heightT, widthT])

    # close down the queue
    queue_coordinator.request_stop()
    queue_coordinator.join(queue_worker_threads)

    # clean up summary file writer
    sfw.close()

print('image height x width : {}x{}'.format(height,width))

'''
    OK, now we can successfully read in TF records.
    
    Notice that everytime we query a property from the TF record
    we evaluate the entire compute chain up through parse_single_example()
    and into reader.read().  This means that we must querey all the data
    from a single TF record in a single call (subsequent calls will trigger 
    additional parse which triggers additional read which triggers additional
    file dequeue).  This can be more easily seen if we break up the calls
    for height and width into two separate calls ...
'''