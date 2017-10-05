import os, tensorflow as tf, numpy as np

from PIL import Image

# the full file path for an image of class A
record_file_path = os.path.join(os.getcwd(), '337975.tfrecord')

# init a file queue (it's just one file)
file_queue = tf.train.string_input_producer([record_file_path], name='file_queue', num_epochs=1)

# init record reader
reader = tf.TFRecordReader()

# read the record file
tfrecord_filename_op, tfrecord_read_op = reader.read(file_queue)

# parse a TF example from serialized byte stream
tfrecord = tf.parse_single_example(
    tfrecord_read_op,
    features={
        'image/object/bbox/xmin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(tf.float32),
        'image/object/label': tf.VarLenFeature(tf.int64),
        'image/encoded': tf.FixedLenFeature([], tf.string)
    },
    name='features'
)

# image was saved as uint8, so we have to decode tf.string as uint8.
imageT = tf.decode_raw(tfrecord['image/encoded'], tf.uint8)

# get the bbox labels for this image
labelsT = tfrecord['image/object/label']

with tf.Session() as sess:
    # init vars
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # init summary file writer
    sfw = tf.summary.FileWriter(os.getcwd(), graph=sess.graph)

    # Start populating the filename queue.
    queue_coordinator = tf.train.Coordinator()
    queue_worker_threads = tf.train.start_queue_runners(coord=queue_coordinator)

    # alternatively reshape tensor into image dimensions
    image, bbox_labels = sess.run([imageT, labelsT])

    # close down the queue
    queue_coordinator.request_stop()
    queue_coordinator.join(queue_worker_threads)

    # clean up summary file writer
    sfw.close()

# we can reshape the image tensor with numpy too
# this is nice b/c it can happen "out of session"
image = np.reshape(image,(480,640,3))

# init PIL image from numpy array
pil_image = Image.fromarray(image)

# show the image
pil_image.show()

# blab about the types of the labels
print('bbox labels type: {}'.format(type(bbox_labels)))

# blab about the labels
print('bbox labels: {}'.format(bbox_labels))

'''
    Now that the image is color, there are three channels.
    Must update the reshape call accordingly.

    OK, one last thing, notice now that we have a 
    SparseTensorValue for the labels ... 
'''