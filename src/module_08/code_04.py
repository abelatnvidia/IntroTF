import os, tensorflow as tf

# the full file path for an image of class A
record_file_path = os.path.join(os.getcwd(),'337975.tfrecord')

# init a file queue (it's just one file)
file_queue = tf.train.string_input_producer([record_file_path],name='file_queue',num_epochs=1)

# init record reader
reader = tf.TFRecordReader()

# read the record file
tfrecord_filename_op, tfrecord_read_op = reader.read(file_queue)

# parse a TF example from serialized byte stream
tfrecord = tf.parse_single_example(
    tfrecord_read_op,
    features={
        'image/object/bbox/xmin': tf.VarLenFeature  (   tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature  (   tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature  (   tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature  (   tf.float32),
        'image/object/label'    : tf.VarLenFeature  (   tf.int64  ),
        'image/encoded'         : tf.FixedLenFeature([],tf.string )
    },
    name='features'
)

# image was saved as uint8, so we have to decode tf.string as uint8.
imageT  = tf.decode_raw(tfrecord['image/encoded'], tf.uint8)

# get the bbox labels for this image
labelsT = tfrecord['image/object/label']

with tf.Session() as sess:

    # init vars
    sess.run(tf.global_variables_initializer())
    sess.run(tf. local_variables_initializer())

    # init summary file writer
    sfw = tf.summary.FileWriter(os.getcwd(),graph=sess.graph)

    # Start populating the filename queue.
    queue_coordinator    = tf.train.Coordinator()
    queue_worker_threads = tf.train.start_queue_runners(coord=queue_coordinator)

    # alternatively reshape tensor into image dimensions
    image,labels = sess.run([imageT,labelsT])

    # close down the queue
    queue_coordinator.request_stop()
    queue_coordinator.join(queue_worker_threads)

    # clean up summary file writer
    sfw.close()

'''
    OK, that's all working great!  Make sure to always do a sanity check ...
'''