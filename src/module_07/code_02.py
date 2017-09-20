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
file_path = os.path.join('data','notMNIST','A','MDEtMDEtMDAudHRm.png')

# get image_buffer (just a bunch of bytes)
with tf.gfile.FastGFile(file_path, 'rb') as f: image_data = f.read()

# decode the binary data as a png image and return a tensor
image_tensor_op = tf.image.decode_png(image_data,channels=1)

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

# pack image metadata, label, and data together as a single training example
example = tf.train.Example(
    features=tf.train.Features(
        feature={
            'image/height'     : _int64_feature(height      ),
            'image/width'      : _int64_feature(width       ),
            'image/colorspace' : _bytes_feature(colorspace  ),
            'image/channels'   : _int64_feature(channels    ),
            'image/class/label': _int64_feature(label       ),
            'image/class/text' : _bytes_feature(text        ),
            'image/format'     : _bytes_feature(image_format),
            'image/filename'   : _bytes_feature(file_name   ),
            'image/encoded'    : _bytes_feature(image_data  )
        }
    )
)

# print out the proto buffer description
print(example)

# what is the type of example
print('the example type: {}'.format(type(example)))

'''
    Very important to notice, that image data is provided by 
    tf.gfile.FastGFile.read() and that no tf.Session() is 
    required to import the bytes from the image file However, 
    see that we actually need tf.Session to decode those bytes!
    If we want to get the image dimensions then we have to lauch 
    session to decodeOtherwise, we just hand off image_data as 
    the image/encoded feature of the example.
    
    You can use numpy and PIL here also without really any difference
    
    This is all pretty mechanical and just protobuf acrobatics
    Encoding TFRecords and Examples will look almost identical every time.
    
    The names of the features such as 'image/width' 'image/encoded' are
    arbitrary in that we can just use whatever we want.  That is, 
    TensorFlow does not mandate what goes in a example or tfrecord
    Basically, TF just thinks of this as a blob of binary data
    which can be passed around to workers/trainers 
'''

