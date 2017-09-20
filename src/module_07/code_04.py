import os, tensorflow as tf

# the full file path for an image of class A
record_file_path = os.path.join(os.getcwd(),'dat.tfrecord')

# init record reader
reader = tf.TFRecordReader()

# read the record file
_, tfrecord_serialized = reader.read(record_file_path)

# # label and image are stored as bytes but could be stored as
# # int64 or float64 values in a serialized tf.Example protobuf.
# tfrecord = tf.parse_single_example(
#     tfrecord_serialized,
#     features={
#         'image/height'     : tf.FixedLenFeature([], tf.int64 ),
#         'image/width'      : tf.FixedLenFeature([], tf.int64 ),
#         'image/colorspace' : tf.FixedLenFeature([], tf.string),
#         'image/channels'   : tf.FixedLenFeature([], tf.int64 ),
#         'image/class/label': tf.FixedLenFeature([], tf.int64 ),
#         'image/class/text' : tf.FixedLenFeature([], tf.string),
#         'image/format'     : tf.FixedLenFeature([], tf.string),
#         'image/filename'   : tf.FixedLenFeature([], tf.string),
#         'image/encoded'    : tf.FixedLenFeature([], tf.string)
#     },
#     name='features'
# )

'''
    oi vey ... we can not pass file path to reader.read (!!)
    
    Can only pass a Queue to reader so need to use string_input_producer 
    
    -\_(``/)_/-
'''