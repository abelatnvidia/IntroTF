import os, tensorflow as tf

# the full file path for tf record file with images
record_file_path = os.path.join(os.getcwd(),'dat.tfrecord')

# init a file path queue
file_queue = tf.train.string_input_producer([record_file_path],name='file_queue',num_epochs=1)

# init record reader
reader = tf.TFRecordReader()

# read the record file
tf_file_name, tfrecord_read_op = reader.read(file_queue)

# init placeholder for serialized record from reader
record = tf.placeholder(dtype=tf.string,name='image-data')

# parse the serialized record into dict
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

# get the label for the image and perform one-hot encoding
label_op = tf.one_hot(tfrecord['image/class/label'],10)

# image was saved as uint8, so we have to decode tf.string as uint8
# lets also reshape the decoded bytes back into the image dimensions
# could also add here: tf.image.resize_image_with_crop_or_pad
image_op = tf.reshape(
              tf.decode_raw(
                  tfrecord['image/encoded'], tf.uint8
              ),
              [28,28]
)

# create batch of images and labels
image_batch_op, label_batch_op = \
    tf.train.shuffle_batch(
                           [image_op, label_op]  ,
                           batch_size       = 32 ,
                           capacity         = 1024,
                           num_threads      = 1  ,
                           min_after_dequeue= 100
                          )


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

    # pull image/label batches from the the tf record
    image_batch,label_batch = sess.run([image_batch_op,label_batch_op])

    # close down the queue
    queue_coordinator.request_stop()
    queue_coordinator.join(queue_worker_threads)

    # clean up summary file writer
    sfw.close()

# blab about the sizes for sanity check
print('image batch size: {}'.format(image_batch.shape))
print('label batch size: {}'.format(label_batch.shape))