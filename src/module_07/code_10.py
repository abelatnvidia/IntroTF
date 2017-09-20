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

# init session for png decoding
sess = tf.Session()

# any name will work (blahblahblah.tfrecord)
tf_record_file_name = 'dat.tfrecord'

# init a TFRecord writer
tf_record_writer = tf.python_io.TFRecordWriter(tf_record_file_name)

# placeholder for image data
image_data = tf.placeholder(dtype=tf.string,name='image-data')

# decode the binary data as a png image and return a tensor
image_tensor_op = tf.image.decode_png(image_data, channels=1)

# init some image/data properties
channels     =  1
image_format = 'PNG'
colorspace   = 'Grayscale'

# create labels/categories
categories = [ 0 , 1 , 2 , 3 , 4 , 5 , 6 , 7 , 8 , 9 ]
labels     = ['A','B','C','D','E','F','G','H','I','J']

# create lookup table to map directory names (i.e. labels) to categroy IDs
label2category = dict(zip(labels,categories))

# initialize record counter
record_count = 0

# walk the data directory for getting data file paths
for root, dir, files in os.walk("./data/notMNIST"):

    # iterate over each file in the directory ...
    for file_name in files:

        # make sure png file
        if not file_name.endswith('.png'): continue

        # create full file path
        file_path = os.path.join(root, file_name)

        # generate comfort signal for humans
        print('processing file: {}'.format(file_path))

        # get image bytes using tf.gfile non-blocking reader
        with tf.gfile.FastGFile(file_path, 'rb') as gfile: image_data_buf = gfile.read()

        # sometimes decode throws exception if fails or empty file etc etc
        try:

            # decode the PNG image data
            image_numpy = sess.run(image_tensor_op,{image_data: image_data_buf})

        except:

            # not much we can do other than blab about it ...
            print('failed to process image: {}'.format(file_path))

            # move on to the next images since decoding failed
            continue

        # compute some meta data about this image
        height    = image_numpy.shape[0]
        width     = image_numpy.shape[1]

        # the human text label for the image category
        image_label = os.path.split(root)[-1]

        # compute the iage category from lookup table
        image_category = label2category[image_label]

        # create "Example" protobuf
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    'image/height'     : _int64_feature(height               ),
                    'image/width'      : _int64_feature(width                ),
                    'image/colorspace' : _bytes_feature(colorspace           ),
                    'image/channels'   : _int64_feature(channels             ),
                    'image/class/label': _int64_feature(image_category       ),
                    'image/class/text' : _bytes_feature(image_label          ),
                    'image/format'     : _bytes_feature(image_format         ),
                    'image/filename'   : _bytes_feature(file_name            ),
                    'image/encoded'    : _bytes_feature(image_numpy.tobytes())
                }
            )
        )

        # finally, add the example to the TF record file
        tf_record_writer.write(example.SerializeToString())

        # update record count
        record_count += 1

# close that session
sess.close()

# be sure to clean up (!)
tf_record_writer.close()

# get the final size of the record file
tf_record_file_size = os.path.getsize(tf_record_file_name)

# another comfort signal for humans
print('created TF record file [{:.2f} MB, {:d} records]: {}'.format(
        tf_record_file_size/1024/1024,record_count,tf_record_file_name
    )
)

'''
    modify this example to create many tf record files with 64 examples each
'''