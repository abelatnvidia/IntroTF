import os, tensorflow as tf

# the full file path for an image of class A
record_file_path = os.path.join(os.getcwd(),'dat.tfrecord')

# init a file queue (i know, i know, it's *just* one file ...)
file_queue = tf.train.string_input_producer([record_file_path],name='file_queue',num_epochs=1)

# init record reader
reader = tf.TFRecordReader()

# read the record file
_, tfrecord_read_op = reader.read(file_queue)

# init placeholder for serialized record from reader
record = tf.placeholder(dtype=tf.string,name='image-data')

# parse the serialized record into dict
tfrecord = tf.parse_single_example(
    record,
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
    sess.run(tf. local_variables_initializer())

    # init summary file writer
    sfw = tf.summary.FileWriter(os.getcwd(),graph=sess.graph)

    # Start populating the filename queue.
    queue_coordinator    = tf.train.Coordinator()
    queue_worker_threads = tf.train.start_queue_runners(coord=queue_coordinator)

    # read record from file
    serialized_record = sess.run(tfrecord_read_op)

    # evaluate height tensor based on serialized record
    height = sess.run(heightT,{record: serialized_record})

    # evalueate width based on same serialized record
    width = sess.run(widthT,{record: serialized_record})

    # get the image data
    image_data = sess.run(imageT,{record: serialized_record})

    # close down the queue
    queue_coordinator.request_stop()
    queue_coordinator.join(queue_worker_threads)

    # clean up summary file writer
    sfw.close()

print('image height x wideth : {}x{}'.format(height,width))
print('image data shape: {}'.format(image_data.shape))

'''
    Here we have refactored the TFReader example to utilize a placeholder
    for the serialized record.  Using a placeholder prevents that new record
    is parsed everytime we access the record.
'''