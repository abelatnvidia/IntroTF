import os, json, tensorflow as tf

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


def _float_feature(value):
    # make sure to always encapsulate as list
    if not isinstance(value, list): value = [value]

    # we're done
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

# coco image id
image_id = '337975'

# define a relative path to the coco annotations and image
json_annotations_file_path = os.path.join('data', 'coco', image_id) + '.json'
coco_image_file_path       = os.path.join('data', 'coco', image_id) + '.jpeg'

# slurp up the data file
with open(json_annotations_file_path) as fdat: data = json.load(fdat)

# blab about the data type
print('imported json annotations as type: {}'.format(type(data)))

# print the number of annotations listed
print('number of annotations imported: {}'.format(len(data['annotations'])))

# init list of bbox coords
xmin = list(); ymin = list(); xmax = list(); ymax = list()

# init list of object labels for each bounding box
labels = list()

# iterate over annotations and generate list of bbox coords
for annotation in data['annotations']:

    # get the category for this bounding box
    labels.append(annotation['category_id'])

    # get the coordinates for this object bbox
    # hm ... the coco format might actually be
    # [xmin, ymin, xmax, ymax]
    ymin.append(annotation['bbox'][0])
    xmin.append(annotation['bbox'][1])
    ymax.append(annotation['bbox'][2])
    xmax.append(annotation['bbox'][3])

# get image_buffer
with tf.gfile.FastGFile(coco_image_file_path, 'rb') as f: image_data = f.read()

# decode the binary data as a jpg image and return a tensor
image_tensor_op = tf.image.decode_jpeg(image_data, channels=3)

# execute the image tensor op to get the numpy array
with tf.Session() as sess: image_numpy = sess.run(image_tensor_op)

print('image_numpy shape: {}'.format(image_numpy.shape))

# pack image metadata, label, and data together as a single training example
example = tf.train.Example(
    features=tf.train.Features(
        feature={
            'image/object/bbox/xmin': _float_feature(xmin),
            'image/object/bbox/xmax': _float_feature(xmax),
            'image/object/bbox/ymin': _float_feature(ymin),
            'image/object/bbox/ymax': _float_feature(ymax),
            'image/object/label'    : _int64_feature(labels),
            'image/encoded'         : _bytes_feature(image_numpy.tobytes())
        }
    )
)

# any name will work
tf_record_file_name = image_id+'.tfrecord'

# init a TFRecord writer
tf_record_writer = tf.python_io.TFRecordWriter(tf_record_file_name)

# serialize the example to a string (bytes) and write
tf_record_writer.write(example.SerializeToString())

# don't forget to clean up (!!)
tf_record_writer.close()

'''
  When working with color image, need/must set the number 
  of channels on the decoder to three.   
'''