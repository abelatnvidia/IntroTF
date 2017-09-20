import os, tensorflow as tf

# the full file path for an image of class A
record_file_path = os.path.join(os.getcwd(),'dat.tfrecord')

# init a file queue (i know, i know, it's *just* one file ...)
file_queue = tf.train.string_input_producer([record_file_path],name='file_queue',num_epochs=1)

# init record reader
reader = tf.TFRecordReader()

# read the record file
_, tfrecord_serialized = reader.read(file_queue)

# label and image are stored as bytes but could be stored as
# int64 or float64 values in a serialized tf.Example protobuf.
tfrecord = tf.parse_single_example(
    tfrecord_serialized,
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

    # evaluate the tensors to get their values
    image, height, width = sess.run([imageT,heightT,widthT])

print('image height x wideth : {}x{}'.format(height,width))

'''
    Why does this hang??
    
    ... don't forget to start the Queue coordinators and thread runners ... 
    
    This is total overkill for a single file, but the general design pattern 
    scales to many many files.  Usually, most modern machine learning projects
    are large and must scale to large "big data".
'''