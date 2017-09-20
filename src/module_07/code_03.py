import os, tensorflow as tf


def _int64_feature(value):

    # make sure to always encapsulate as list
    if not isinstance(value, list): value = [value]

    # we're done
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    # make sure convert to bytes same python 2 as python 3
    value = tf.compat.as_bytes(value)

    # we're done
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# the full file path for an image of class A
file_path = os.path.join('data', 'notMNIST', 'A', 'MDEtMDEtMDAudHRm.png')

# get image_buffer (just a bunch of bytes)
with tf.gfile.FastGFile(file_path, 'rb') as f: image_data = f.read()

# decode the binary data as a png image and return a tensor
image_tensor_op = tf.image.decode_png(image_data, channels=1)

# execute the image tensor op to get the numpy array
with tf.Session() as sess: image_numpy = sess.run(image_tensor_op)

# compile some metadata for this image
label        =  0
text         = 'A'
channels     =  1
image_format = 'PNG'
colorspace   = 'Grayscale'
height       = image_numpy.shape[0]
width        = image_numpy.shape[1]
file_name    = os.path.basename(file_path)

print('image_data  type: {}'.format(type(image_data)))
print('image_numpy type: {}'.format(image_numpy.dtype))

# pack image metadata, label, and data together as a single training example
example = tf.train.Example(
    features=tf.train.Features(
        feature={
            'image/height'     : _int64_feature(height               ),
            'image/width'      : _int64_feature(width                ),
            'image/colorspace' : _bytes_feature(colorspace           ),
            'image/channels'   : _int64_feature(channels             ),
            'image/class/label': _int64_feature(label                ),
            'image/class/text' : _bytes_feature(text                 ),
            'image/format'     : _bytes_feature(image_format         ),
            'image/filename'   : _bytes_feature(file_name            ),
            'image/encoded'    : _bytes_feature(image_numpy.tobytes())
        }
    )
)

# any name will work (blahblahblah.record)
tf_record_file_name = 'dat.tfrecord'

# init a TFRecord writer
tf_record_writer = tf.python_io.TFRecordWriter(tf_record_file_name)

# serialize the example to a string (bytes) and write
tf_record_writer.write(example.SerializeToString())

# be sure to clean up (!)
tf_record_writer.close()

'''
    Complete this example and exports it as TFRecord.
    
    If you have more than a single example, you can write many 
    to single TFRecord file.  Sometimes, if there are many many files, 
    it is common practice to "shard" the examples across a many TFRecord files
    That is, if there are 1.4M images you might consider creating labeled
    examples and writing 256 examples per record.  This makes it a bit easier 
    to manage rather than single huge record file that has to be loaded all 
    at once etc.
'''

